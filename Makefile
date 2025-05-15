ruff:
	ruff check ./src ./app ./tests 
	ruff check --fix ./src ./app ./tests 
	ruff format ./src ./app ./tests 