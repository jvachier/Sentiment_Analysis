ruff:
	ruff check ./modules sentiment_analysis.py
	ruff check --fix ./modules sentiment_analysis.py
	ruff format ./modules sentiment_analysis.py