import unittest
from sklearn.datasets import make_classification
from train_model import train_model

class TestTrainModel(unittest.TestCase):
    def test_train_model(self):
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=24)
        clf, X_test, y_test = train_model(X, y)
        self.assertIsNotNone(clf)
        
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_test), 20)
        
        predictions = clf.predict(X_test)
        self.assertEqual(len(predictions), len(y_test))

if __name__ == '__main__':
    unittest.main()