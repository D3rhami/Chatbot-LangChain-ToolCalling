from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from process import get_system_knowledge


class MemoryManager:
    """Manages conversation memory for users."""

    def __init__(self):
        self.memories = {}

    def get_memory(self, user_id: str) -> ChatMessageHistory:
        """Get or create memory for a user."""
        if user_id not in self.memories:
            history = ChatMessageHistory()
            # Add system message with knowledge
            sk=get_system_knowledge()
            system_message = SystemMessage(content=sk)
            history.add_message(system_message)
            self.memories[user_id] = history
        return self.memories[user_id]

    def clear_memory(self, user_id: str) -> bool:
        """Clear memory for a user but keep system message."""
        if user_id in self.memories:
            system_content = get_system_knowledge()
            history = ChatMessageHistory()
            history.add_message(SystemMessage(content=system_content))
            self.memories[user_id] = history
            return True
        return False

    def add_human_message(self, user_id: str, content: str) -> None:
        """Add a human message to memory."""
        memory = self.get_memory(user_id)
        memory.add_message(HumanMessage(content=content))

    def add_ai_message(self, user_id: str, content: str) -> None:
        """Add an AI message to memory."""
        memory = self.get_memory(user_id)
        memory.add_message(AIMessage(content=content))

    def get_messages(self, user_id: str) -> list:
        """Get all messages for a user."""
        memory = self.get_memory(user_id)
        return memory.messages

    def prune_memory(self, user_id: str, max_messages: int = 10) -> None:
        """Prune memory to keep a maximum number of messages."""
        memory = self.get_memory(user_id)
        messages = memory.messages

        if len(messages) > max_messages:
            # Keep system message and last (max_messages-1) exchanges
            new_history = ChatMessageHistory()
            # Find the system message
            system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]

            if system_messages:
                new_history.add_message(system_messages[0])

            # Add most recent messages up to the limit
            remaining_slots = max_messages - len(system_messages)
            for msg in messages[-remaining_slots:]:
                if not isinstance(msg, SystemMessage):
                    new_history.add_message(msg)

            self.memories[user_id] = new_history
