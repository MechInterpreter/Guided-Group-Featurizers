"""Apply the fixed, label-free DINOv3 UMAP to a complete image's patch grid."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np


class FrozenUmap:
    def __init__(self, private: Path, public_map_path: Path | None = None):
        receipt = json.loads((private / 'SHARED_UMAP_VERIFIED.json').read_text(encoding='utf-8'))
        model_path = private / 'shared_umap.joblib'
        with model_path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        with (private / 'normalization.pt').open('rb') as stream:
            normalization_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        public_map = json.loads((public_map_path or private.parents[1] / 'public/data/shared-umap.json').read_text(encoding='utf-8'))
        if (not receipt['passed'] or receipt['labels_used'] or digest != receipt['model_sha256']
                or normalization_sha != public_map['normalization_sha256']
                or digest != public_map['model_sha256']):
            raise RuntimeError('Shared UMAP verification failed')
        self.identity = {'model_sha256': digest, 'fit_features_sha256': receipt['fit_features_sha256'],
                         'parameters': receipt['parameters']}
        self.reducer = joblib.load(model_path)
        self.reducer.transform(self.reducer._raw_data[:196].copy())

    def transform(self, features):
        values = np.ascontiguousarray(features, dtype=np.float32)
        if values.shape != (196, 768) or not np.isfinite(values).all():
            raise ValueError('Invalid DINOv3 patch grid for UMAP')
        xyz = self.reducer.transform(values)
        if xyz.shape != (196, 3) or not np.isfinite(xyz).all():
            raise RuntimeError('Invalid shared UMAP coordinates')
        return {'positions': xyz.round(6).tolist(),
                'feature_sha256': hashlib.sha256(values.tobytes()).hexdigest(), **self.identity}
