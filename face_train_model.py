from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

EMB_PATH = 'output/embeddings.pickle'
REC_PATH = 'output/recognizer.pickle'
LE_PATH  = 'output/le.pickle'

print("Loading embeddings...")
data = pickle.loads(open(EMB_PATH, 'rb').read())

print("Encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data['names'])

print('Training model...')
rec = SVC(C=1.0, kernel='linear', probability=True)
rec.fit(data['embeddings'], labels)

with open(REC_PATH, 'wb') as f:
    f.write(pickle.dumps(rec))

with open(LE_PATH, 'wb') as f:
    f.write(pickle.dumps(le))