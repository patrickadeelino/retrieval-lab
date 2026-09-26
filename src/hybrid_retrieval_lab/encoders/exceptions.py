class TokenLimitError(ValueError):
    def __init__(self, model: str, input_kind: str, token_count: int, max_tokens: int) -> None:
        self.model = model
        self.input_kind = input_kind
        self.token_count = token_count
        self.max_tokens = max_tokens
        super().__init__(f"{input_kind} has {token_count} tokens for {model}, exceeding the {max_tokens}-token limit")
