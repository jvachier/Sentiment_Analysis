ruff:
	ruff check ./modules sentiment_analysis.py voice_to_text_app.py
	ruff check --fix ./modules sentiment_analysis.py voice_to_text_app.py
	ruff format ./modules sentiment_analysis.py voice_to_text_app.py