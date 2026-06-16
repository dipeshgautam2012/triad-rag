"""Real answers via Google Gemini (env.toml: ``default_provider = "google"``)."""

import json
from typing import Any

from fastapi import HTTPException

from app.ai_providers.base_provider import BaseAIProvider


class GeminiProvider(BaseAIProvider):
    """Call the Gemini API with the system prompt from prompts.toml and retrieved context."""

    def generate(
        self,
        question: str,
        context: str,
        output_schema: Any | None = None,
    ) -> Any:
        api_key = self._cfg.api_key.strip()
        if not api_key:
            raise HTTPException(status_code=500, detail="GEN_API_KEY is empty for provider=google")

        model = self._cfg.model.strip()
        if not model:
            raise HTTPException(status_code=500, detail="model is empty for provider=google")
        user_prompt = self._user_content(question, context)
        system = self._system_instruction()
        gen_config: dict[str, Any] = {"system_instruction": system}

        try:
            import google.genai as genai  # type: ignore[import-not-found]

            client = genai.Client(api_key=api_key)
            if output_schema is None:
                response = client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=gen_config,
                )
                text = str(response.text or "").strip()
                if not text:
                    raise HTTPException(status_code=502, detail="google generation returned empty answer")
                return text

            gen_config["response_mime_type"] = "application/json"
            gen_config["response_schema"] = output_schema
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=gen_config,
            )
            raw = str(response.text or "").strip()
            if not raw:
                raise HTTPException(status_code=502, detail="google generation returned empty answer")
            return json.loads(raw)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"google generation failed: {e!s}") from e
