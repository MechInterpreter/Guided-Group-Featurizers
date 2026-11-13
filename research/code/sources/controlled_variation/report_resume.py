"""Keep verified cache identities across this report-only repair."""
from pathlib import Path
import json,re

# These prior bundles have the same scientific analysis. This repair changes
# the point-report seed fields and checkpoint lookup, not cached computations.
COMPATIBLE_PREFIXES=('2c115474fe6c54c9','f1670ac0adee1773')

def cache_identity(output, execution_sha):
    if not re.fullmatch('[0-9a-f]{64}',execution_sha):
        raise ValueError('Invalid execution source checksum')
    candidates=[]
    for p in sorted((Path(output)/'analysis/results').glob('*/protocol.json')):
        q=json.loads(p.read_text())
        value=q.get('source_sha','')
        if q.get('version')!=3 or not q.get('semantic_protocol','').startswith('five native methods'):
            continue
        if re.fullmatch('[0-9a-f]{64}',value):
            candidates.append(value)
    # Only the source salt is reused. The complete protocol still hashes the
    # actual model weights, cohort, normalization and native model identities.
    for prefix in COMPATIBLE_PREFIXES:
        hits=sorted({v for v in candidates if v.startswith(prefix)})
        if len(hits)>1:
            raise ValueError('Ambiguous prior bundle checksum for '+prefix)
        if hits:
            return hits[0]
    return execution_sha
