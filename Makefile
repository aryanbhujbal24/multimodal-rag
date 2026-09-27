.PHONY: install corpus index serve ui test eval sweep docker

install:
	pip install -r requirements.txt

corpus:          ## download the arXiv demo corpus
	python scripts/fetch_corpus.py

index:           ## build the FAISS index (with VLM figure descriptions)
	python -m scripts.build_index

index-fast:      ## build without VLM descriptions (no API calls)
	python -m scripts.build_index --no-describe

serve:
	uvicorn app.api.main:app --reload --port 8000

ui:
	streamlit run app/ui/streamlit_app.py

test:
	EMBEDDER=hashing pytest tests -q

eval:
	python -m eval.evaluate --questions eval/questions.jsonl --generation

sweep:
	python -m eval.evaluate --questions eval/questions.jsonl --sweep

docker:
	docker compose up --build
