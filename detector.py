"""InsightFace face detection and embedding comparison."""
import numpy as np
from insightface.app import FaceAnalysis


class FaceDuplicateDetector:
    def __init__(self, model_name="buffalo_l", providers=None):
        providers = providers or ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name=model_name, providers=providers)
        # CPU-friendly default. InsightFace may download model weights on first startup.
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def detect_faces(self, image_bgr):
        """Return normalized embeddings and bounding boxes for faces in a BGR image."""
        faces = self.app.get(image_bgr)
        output = []
        for face in faces:
            embedding = np.asarray(face.normed_embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm == 0:
                continue
            embedding = embedding / norm
            bbox = [int(round(float(x))) for x in face.bbox]
            output.append({"embedding": embedding, "bbox": bbox})
        return output

    @staticmethod
    def best_similarity(embeddings_a, embeddings_b):
        """Highest cosine similarity across any face pair in two images, or None."""
        if not embeddings_a or not embeddings_b:
            return None
        a = np.vstack(embeddings_a).astype(np.float32)
        b = np.vstack(embeddings_b).astype(np.float32)
        # Embeddings are normalized, so dot product equals cosine similarity.
        similarities = a @ b.T
        return float(np.max(similarities))
