from setuptools import setup, find_packages

setup(
    name="eth-algo-trading",
    version="0.1.0",
    description="AI-powered algorithmic trading strategies for Ethereum",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "hmmlearn>=0.3.0",
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "ccxt>=4.0.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
    ],
)
