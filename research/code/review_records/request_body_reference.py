"""Exact original request construction. For protocol inspection; no requests sent."""
import base64,io,json
from PIL import Image
from rubric_validation import *
PROVIDER="google-ai-studio"
PAID_PROVIDER="google-ai-studio"
STRONGER_PROVIDER="google-ai-studio/flex"
def encode(value): return json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8')

def request_body(case, template, rubric, card, family, *, paid=False, stronger=False, concise=False):
    # From prepare_evaluation_reviews.card: six 224px columns. The crops
    # include the numbered views, their marks and every zoom strip.
    with Image.open(io.BytesIO(card)) as image:
        require(image.size == (1344, 364), 'Unexpected card geometry')
        images = [card]
        for start in (0, 448, 896):
            crop = image.crop((start, 28, start + 448, 364))
            buf = io.BytesIO()
            crop.save(buf, format='PNG')
            images.append(buf.getvalue())
    explanation = (
        'Use the rubric above. This API call is one fresh context for one case. '
        'The first image is the full card. The next three show views 1–2, 3–4, and 5–6 '
        'with their zoom strips at the original pixel scale. They repeat the same views. '
        'Return exactly one JSON object matching the case template below. '
        'The local runner adds reviewer identity, provenance and file storage.'
    )
    if family == 'pair':
        explanation += (
            ' Pair IDs 0, 1 and 2 correspond to views 1–2, 3–4 and 5–6. '
            'For every pair, give a concrete meaning or explain the exception. '
            'If marked detail is unreadable, keep the relevant judgment unclear.'
        )
    else:
        explanation += ' Format each exception as {"view": 1, "reason": "concrete explanation"}.'
    if paid:
        explanation += (
            ' Inspect each numbered marked region and its zoom before deciding. '
            'A recognizable object in the surrounding photo is insufficient evidence. '
            'Only name an exact letter, part or texture as shared if the marks in both views support it. '
            'Describe what the marks actually show. Where they differ or are unreadable, explain that uncertainty.'
        )
        if family == 'pair':
            explanation += ' For each pair, cite the visible evidence in both numbered views in shared_meaning or exceptions.'
        else:
            explanation += ' Fill observations with concrete support from the numbered marked views, even when every view fits.'
    if stronger:
        explanation += (
            ' The pink squares, numbering, borders and layout are review overlays. '
            'They never count as a shared feature or theme. Judge the image content beneath the squares '
            'using the zoom strips. Scene context can help identify the content within the marks.'
        )
        if family == 'pair':
            explanation += (
                ' A visual yes needs a specific common visual feature supported in both sets of marked regions. '
                'If the marked appearances differ and there is no such feature, record visual=no. '
                'If evidence is unreadable, record unclear. State the actual shared feature for each yes.'
            )
        else:
            explanation += (
                ' Choose the fitting valid views first, then apply the frozen fit-count rule. '
                'Use one theme across those views and record an exception for every other valid view.'
            )
    if concise:
        if family == 'pair':
            explanation += (
                ' Judge semantic and visual coherence independently. A specifically named shared part, '
                'material or region qualifies semantically when visible in both sets of marks, even '
                'if the parent objects, actions or settings differ. Use scene context only to identify '
                'the marked content. If your own evidence establishes such a match, record semantic=yes. '
                'Keep shared_meaning and exceptions to at most 8 words each. Use an empty string '
                'for an inapplicable field. A yes on either axis needs a short concrete shared feature.'
            )
        else:
            explanation += (
                ' Keep theme to at most 8 words, observations to at most 12 words, and each '
                'exception reason to at most 5 words. Record each nonmatching valid view.'
            )
        explanation += ' Use compact JSON and follow the supplied JSON schema.'
    text = f'{rubric}\n\n{explanation}\n\nCase metadata\n{json.dumps(case)}'
    if not concise:
        text += f'\n\nCase template\n{json.dumps(template)}'
    content = [{'type': 'text', 'text': text}]
    for raw in images:
        content.append({'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,' + base64.b64encode(raw).decode('ascii')}})
    body = {
        'model': MODEL,
        'provider': {'only': [PROVIDER], 'allow_fallbacks': False},
        'temperature': 0,
        'max_tokens': 2048,
        'response_format': {'type': 'json_object'},
        'messages': [{'role': 'user', 'content': content}],
    }
    if paid:
        body['model'] = PAID_MODEL
        body['provider'] = {
            'only': [PAID_PROVIDER],
            'allow_fallbacks': False,
            'max_price': {'prompt': 0.10, 'completion': 0.40},
        }
        body['response_format'] = {
            'type': 'json_schema',
            'json_schema': {'name': f'blinded_{family}_review', 'strict': True,
                            'schema': paid_schema(case, family)},
        }
        body['usage'] = {'include': True}
    if stronger:
        body['model'] = STRONGER_MODEL
        body['provider']['only'] = [STRONGER_PROVIDER]
        body['provider']['max_price'] = {'prompt': 0.15, 'completion': 1.25}
        body['service_tier'] = 'flex'
        body['reasoning'] = {'effort': 'minimal'}
    return encode(body)
