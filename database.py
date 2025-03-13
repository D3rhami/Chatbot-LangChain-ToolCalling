# Standard library imports
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
# SQLAlchemy imports
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

# Create engine
try:
    engine = create_engine(DATABASE_URL)
    logger.info("Database engine created successfully")
except Exception as e:
    logger.error(f"Error creating database engine: {str(e)}")
    raise


def get_database_schema() -> dict[
    str, dict[str, list[dict[str, str | bool]] | list[dict[str, str | list[str]]] | list[str]]]:
    """
    Get the schema of the database to dynamically adapt to any changes.

    Returns:
        Dict containing tables and their columns
    """
    try:
        inspector = inspect(engine)
        schema = {}

        for table_name in inspector.get_table_names():
            columns = []
            for column in inspector.get_columns(table_name):
                columns.append({
                        "name": column["name"],
                        "type": str(column["type"]),
                        "nullable": column["nullable"]
                })

            # Get foreign keys
            foreign_keys = []
            for fk in inspector.get_foreign_keys(table_name):
                foreign_keys.append({
                        "referred_table": fk["referred_table"],
                        "referred_columns": fk["referred_columns"],
                        "constrained_columns": fk["constrained_columns"]
                })

            # Get primary keys
            primary_keys = inspector.get_pk_constraint(table_name)["constrained_columns"]

            schema[table_name] = {
                    "columns": columns,
                    "foreign_keys": foreign_keys,
                    "primary_keys": primary_keys
            }

        logger.info(f"Retrieved database schema with {len(schema)} tables")
        return schema
    except Exception as error1:
        logger.error(f"Error getting database schema: {str(error1)}")
        raise


def check_order_status(order_id: int) -> Dict[str, Any]:
    """
    Check the status of an order.

    Args:
        order_id: The ID of the order to check

    Returns:
        Dict containing order details and status
    """
    try:
        # Map status codes to human-readable values
        status_map = {
                0: "Pending",
                1: "Shipped",
                2: "Delivered",
                3: "Canceled"
        }

        with engine.connect() as conn:
            # Query order information
            order_query = text("""
                SELECT o.id, o.status, c.name as customer_name, c.phone_number
                FROM orders o
                JOIN customers c ON o.customer_id = c.id
                WHERE o.id = :order_id
            """)
            order_result = conn.execute(order_query, {"order_id": order_id}).fetchone()

            if not order_result:
                logger.warning(f"Order with ID {order_id} not found")
                return {"error": f"Order with ID {order_id} not found"}

            # Get order items
            items_query = text("""
                SELECT p.id, p.name, p.price
                FROM order_product op
                JOIN product p ON op.product_id = p.id
                WHERE op.order_id = :order_id
            """)
            items_result = conn.execute(items_query, {"order_id": order_id}).fetchall()

            # Prepare response
            order_details = {
                    "order_id": order_result.id,
                    "status": status_map.get(order_result.status, "Unknown"),
                    "status_code": order_result.status,
                    "customer": {
                            "name": order_result.customer_name,
                            "phone_number": order_result.phone_number
                    },
                    "items": [
                            {
                                    "product_id": item.id,
                                    "product_name": item.name,
                                    "price": float(item.price)
                            } for item in items_result
                    ],
                    "total_price": sum(float(item.price) for item in items_result)
            }

            logger.info(f"Retrieved order status for order {order_id}: {order_details['status']}")
            return order_details
    except SQLAlchemyError as ch_e1:
        logger.error(f"Database error checking order status: {str(ch_e1)}")
        return {"error": f"Database error: {str(ch_e1)}"}
    except Exception as ch_e2:
        logger.error(f"Error checking order status: {str(ch_e2)}")
        return {"error": f"Error: {str(ch_e2)}"}


def register_order(customer_name: str, phone_number: str, products: List[int] = None) -> Dict[str, Any]:
    """
    Register a new order for a customer.

    Args:
        customer_name: The name of the customer
        phone_number: The phone number of the customer
        products: List of product IDs to include in the order (optional)

    Returns:
        Dict containing the result of the order registration
    """
    if not products:
        products = []

    try:
        with engine.connect() as conn:
            # Start transaction
            trans = conn.begin()
            try:
                # Check if customer exists, if not create
                customer_query = text("""
                    SELECT id FROM customers 
                    WHERE phone_number = :phone_number
                """)
                customer_result = conn.execute(customer_query, {"phone_number": phone_number}).fetchone()

                if customer_result:
                    customer_id = customer_result.id
                    # Update name if it's different
                    update_query = text("""
                        UPDATE customers 
                        SET name = :customer_name 
                        WHERE id = :customer_id
                    """)
                    conn.execute(update_query, {
                            "customer_name": customer_name,
                            "customer_id": customer_id
                    })
                    logger.info(f"Updated existing customer: {customer_id} - {customer_name}")
                else:
                    # Create new customer
                    insert_query = text("""
                        INSERT INTO customers (name, phone_number) 
                        VALUES (:customer_name, :phone_number) 
                        RETURNING id
                    """)
                    customer_id = conn.execute(insert_query, {
                            "customer_name": customer_name,
                            "phone_number": phone_number
                    }).fetchone().id
                    logger.info(f"Created new customer: {customer_id} - {customer_name}")

                # Create order
                order_query = text("""
                    INSERT INTO orders (customer_id, status) 
                    VALUES (:customer_id, 0) 
                    RETURNING id
                """)
                order_id = conn.execute(order_query, {"customer_id": customer_id}).fetchone().id
                logger.info(f"Created new order: {order_id}")

                # Add products to order if provided
                added_products = []

                for product_id in products:
                    # Check if product exists and has stock
                    product_query = text("""
                        SELECT id, name, price, stock_quantity 
                        FROM product 
                        WHERE id = :product_id
                    """)
                    product_result = conn.execute(product_query, {"product_id": product_id}).fetchone()

                    if not product_result:
                        logger.warning(f"Product {product_id} not found")
                        continue

                    if product_result.stock_quantity <= 0:
                        logger.warning(f"Product {product_id} out of stock")
                        continue

                    # Add product to order
                    order_product_query = text("""
                        INSERT INTO order_product (order_id, product_id) 
                        VALUES (:order_id, :product_id)
                    """)
                    conn.execute(order_product_query, {
                            "order_id": order_id,
                            "product_id": product_id
                    })

                    # Reduce stock quantity
                    update_stock_query = text("""
                        UPDATE product 
                        SET stock_quantity = stock_quantity - 1 
                        WHERE id = :product_id
                    """)
                    conn.execute(update_stock_query, {"product_id": product_id})

                    added_products.append({
                            "product_id": product_result.id,
                            "product_name": product_result.name,
                            "price": float(product_result.price)
                    })

                # Commit transaction
                trans.commit()

                return {
                        "order_id": order_id,
                        "customer": {
                                "id": customer_id,
                                "name": customer_name,
                                "phone_number": phone_number
                        },
                        "status": "Pending",
                        "products": added_products,
                        "total_price": sum(product["price"] for product in added_products)
                }
            except Exception as rg_e:
                # Rollback transaction in case of error
                trans.rollback()
                logger.error(f"Error in transaction, rolling back: {str(rg_e)}")
                raise
    except SQLAlchemyError as rg_e1:
        logger.error(f"Database error registering order: {str(rg_e1)}")
        return {"error": f"Database error: {str(rg_e1)}"}
    except Exception as rg_e2:
        logger.error(f"Error registering order: {str(rg_e2)}")
        return {"error": f"Error: {str(rg_e2)}"}


def check_inventory(product_id: Optional[int] = None, product_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Check inventory for a product by ID or name, or get all inventory.

    Args:
        product_id: The ID of the product to check (optional)
        product_name: The name of the product to check (optional)

    Returns:
        Dict containing inventory information
    """
    try:
        with engine.connect() as conn:
            if product_id is not None:
                # Query by product ID
                query = text("""
                    SELECT id, name, price, stock_quantity 
                    FROM product 
                    WHERE id = :product_id
                """)
                result = conn.execute(query, {"product_id": product_id}).fetchone()

                if not result:
                    logger.warning(f"Product with ID {product_id} not found")
                    return {"error": f"Product with ID {product_id} not found"}

                return {
                        "product": {
                                "id": result.id,
                                "name": result.name,
                                "price": float(result.price),
                                "stock_quantity": result.stock_quantity,
                                "in_stock": result.stock_quantity > 0
                        }
                }

            elif product_name is not None:
                # Query by product name (using LIKE for partial matching)
                query = text("""
                    SELECT id, name, price, stock_quantity 
                    FROM product 
                    WHERE name ILIKE :product_name
                """)
                results = conn.execute(query, {"product_name": f"%{product_name}%"}).fetchall()

                if not results:
                    logger.warning(f"No products found matching '{product_name}'")
                    return {"error": f"No products found matching '{product_name}'"}

                products = [
                        {
                                "id": result.id,
                                "name": result.name,
                                "price": float(result.price),
                                "stock_quantity": result.stock_quantity,
                                "in_stock": result.stock_quantity > 0
                        } for result in results
                ]

                return {
                        "products": products,
                        "count": len(products)
                }

            else:
                # Get all products
                query = text("""
                    SELECT id, name, price, stock_quantity 
                    FROM product 
                    ORDER BY id
                """)
                results = conn.execute(query).fetchall()

                products = [
                        {
                                "id": result.id,
                                "name": result.name,
                                "price": float(result.price),
                                "stock_quantity": result.stock_quantity,
                                "in_stock": result.stock_quantity > 0
                        } for result in results
                ]

                return {
                        "products": products,
                        "count": len(products),
                        "total_items_in_stock": sum(product["stock_quantity"] for product in products)
                }
    except SQLAlchemyError as er_sql:
        logger.error(f"Database error checking inventory: {str(er_sql)}")
        return {"error": f"Database error: {str(er_sql)}"}
    except Exception as error2:
        logger.error(f"Error checking inventory: {str(error2)}")
        return {"error": f"Error: {str(error2)}"}
