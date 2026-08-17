from __future__ import annotations

import re
from typing import List, Set, Union


class SecurityShield:
    """
    Security Shield & Prompt Injection Guard.
    
    Protects LLMs against Special Token Smuggling and Delimiter Injection Attacks.
    If untrusted user input contains raw control tokens (e.g., '<|endoftext|>', '<|system|>'),
    this module prevents control-plane hijacking by escaping or raising security exceptions.
    """

    def __init__(
        self,
        special_tokens: List[str],
        special_token_pattern: str = r"<\|[^\s|]+\|>",
    ):
        self.special_tokens = set(special_tokens)
        self.pattern = re.compile(special_token_pattern)

    def sanitize(
        self,
        text: str,
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",  # "escape" | "raise" | "ignore"
    ) -> str:
        """
        Sanitizes input text according to allowed special token policy.
        
        - allowed_special="all": all special tokens in text are treated as control tokens.
        - allowed_special="none": all special tokens in text are disallowed/escaped.
        - allowed_special={"<|user|>"}: only specified tokens are preserved as control tokens.
        """
        if allowed_special == "all":
            allowed_set = self.special_tokens
        elif allowed_special == "none" or not allowed_special:
            allowed_set = set()
        else:
            allowed_set = set(allowed_special)

        def replace_fn(match: re.Match) -> str:
            token = match.group(0)
            if token in allowed_set:
                return token  # Keep as authorized control token

            # Token is unauthorized / smuggled in user input
            if disallowed_special_action == "raise":
                raise ValueError(
                    f"Security Exception: Untrusted input attempted to inject control token {token!r}. "
                    f"If this was intentional, pass allowed_special={{{token!r}}}."
                )
            elif disallowed_special_action == "escape":
                # Neutralize control token into safe visible text (e.g. <\|endoftext\|>)
                return f"<\\|{token[2:-2]}\\|>"
            elif disallowed_special_action == "ignore":
                # Drop token entirely
                return ""
            return token

        return self.pattern.sub(replace_fn, text)
