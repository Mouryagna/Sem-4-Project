from src.utils import load_object

pre = load_object("artifacts/preprocessor.pkl")
print(pre.named_transformers_["cat"]["ohe"].categories_)