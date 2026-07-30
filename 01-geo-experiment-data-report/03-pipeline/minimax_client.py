"""MiniMax LLM provider for the citation-analysis pipeline.

The website-categorization step previously relied on a single LLM provider
(hard-coded ``gemini-2.5-flash``). This module adds a MiniMax client so the
same SEO categorization can be served by MiniMax instead. The client targets
the OpenAI-compatible Chat Completions endpoint and supports both the global
(``api.minimax.io``) and China (``api.minimaxi.com``) base URLs, selectable via
``MINIMAX_REGION``. The model is selectable between ``MiniMax-M3`` and
``MiniMax-M2.7`` via ``MINIMAX_MODEL``.

The OpenAI SDK is imported lazily, so importing this module never fails even
when the SDK is not installed; :func:`build_minimax_client` simply returns
``None`` and the pipeline falls back to its default provider.
"""

import os

# Regional endpoints: global (api.minimax.io) and China (api.minimaxi.com) base
# URLs for the OpenAI-compatible Chat Completions API.
MINIMAX_BASE_URLS = {
    "global_en": "https://api.minimax.io/v1",
    "cn_zh": "https://api.minimaxi.com/v1",
}

# Selectable MiniMax text models.
MINIMAX_MODELS = ("MiniMax-M3", "MiniMax-M2.7")

DEFAULT_REGION = "global_en"
DEFAULT_MODEL = "MiniMax-M3"


def resolve_base_url(region=None):
    """Return the OpenAI-compatible base URL for the given region."""
    region = (region or os.environ.get("MINIMAX_REGION", "")).strip() or DEFAULT_REGION
    return MINIMAX_BASE_URLS.get(region, MINIMAX_BASE_URLS[DEFAULT_REGION])


def resolve_model(model=None):
    """Return a selectable MiniMax model, falling back to the default."""
    model = (model or os.environ.get("MINIMAX_MODEL", "")).strip() or DEFAULT_MODEL
    return model if model in MINIMAX_MODELS else DEFAULT_MODEL


def build_minimax_client(api_key=None, region=None, model=None):
    """Build an OpenAI-compatible MiniMax client.

    Returns a ``(client, model)`` tuple, or ``None`` when the API key is
    absent or the OpenAI SDK is unavailable, so callers can degrade to their
    default provider.
    """
    api_key = (api_key or os.environ.get("MINIMAX_API_KEY", "")).strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None
    base_url = resolve_base_url(region)
    resolved_model = resolve_model(model)
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, resolved_model


def categorize_website(domain, minimax_client=None):
    """Classify a domain into an SEO category using MiniMax.

    ``minimax_client`` is the ``(client, model)`` tuple returned by
    :func:`build_minimax_client`. Returns ``"null"`` when no client is
    available, mirroring the existing categorization contract.
    """
    if not minimax_client:
        return "null"
    client, model = minimax_client
    prompt = (
        f"你是SEO分析师。判断域名 {domain} 属于以下哪一类："
        f"新闻 / blog / 行业垂类 / 测评类 / 官网 / 电商 / 其他。"
        f"只能返回一个词。如果无法判断，请回复 null。"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        result = (response.choices[0].message.content or "").strip()
        return result if result else "null"
    except Exception as e:
        print(f"  [MiniMax 异常] {domain}: {e}")
        return "null"
