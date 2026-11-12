# Reproduction of https://unemploymentlabs.com/checkpoint-forensics
# Weights are fetched by HTTP range request (embedding tables only, ~1-2.5 GB each).
DATA ?= data
RUN  := uv run -q python -I -m
SMS  := --sms-range 100 160

.PHONY: test check fetch scan blind
test:
	uv run -q pytest -q
check:
	uv run -q pyright forensics

fetch:
	$(RUN) forensics.fetch_tensor deepseek-ai/DeepSeek-V3-Base model-00001-of-000163.safetensors model.embed_tokens.weight $(DATA)/dsv3/embed.bin
	$(RUN) forensics.fetch_tensor deepseek-ai/DeepSeek-V4-Flash model-00001-of-00046.safetensors embed.weight $(DATA)/dsv4flash/embed.bin
	$(RUN) forensics.fetch_tensor deepseek-ai/DeepSeek-V4-Pro model-00001-of-00064.safetensors embed.weight $(DATA)/dsv4pro/embed.bin
	$(RUN) forensics.fetch_tensor moonshotai/Kimi-K3 model-00094-of-000096.safetensors language_model.model.embed_tokens.weight $(DATA)/k3/embed.bin
	$(RUN) forensics.fetch_tensor moonshotai/Kimi-K3.1-Pro model-00094-of-000096.safetensors language_model.model.embed_tokens.weight $(DATA)/k31/embed.bin
	$(RUN) forensics.fetch_tensor MiniMaxAI/MiniMax-M3 model-00001-of-00059.safetensors language_model.model.embed_tokens.weight $(DATA)/m3/embed.bin
	$(RUN) forensics.fetch_tensor inclusionAI/Ling-3.1-flash model-00001-of-00054.safetensors model.word_embeddings.weight $(DATA)/ling31/embed.bin

# Score every SM count 100..160 for each model's seed (expect one peak each).
scan:
	$(RUN) forensics.scan $(DATA)/dsv3/embed.bin --seed 953049583 $(SMS)
	$(RUN) forensics.scan $(DATA)/dsv4flash/embed.bin --seed 953049583 $(SMS)
	$(RUN) forensics.scan $(DATA)/dsv4pro/embed.bin --seed 953049583 $(SMS)
	$(RUN) forensics.scan $(DATA)/k3/embed.bin --seed 20256586 --offset-calls 944 $(SMS)
	$(RUN) forensics.scan $(DATA)/k31/embed.bin --seed 20256586 --offset-calls 944 $(SMS)
	$(RUN) forensics.scan $(DATA)/m3/embed.bin --seed 2760 $(SMS)
	$(RUN) forensics.scan $(DATA)/ling31/embed.bin --seed 3952 --dist trunc_normal3 --preserved-r 0.95 $(SMS)

# Recover seeds from the weights alone: every 32-bit seed x layouts 108/132/148/152.
blind:
	$(RUN) forensics.seed_search $(DATA)/dsv4flash/embed.bin --rows 128850 128851 128852 128853 --std 0.006
	$(RUN) forensics.seed_search $(DATA)/m3/embed.bin --rows 245 246 247 248 249 250 251 252 --tol-rel 0.04 --tol-abs 0.01
	$(RUN) forensics.seed_search $(DATA)/ling31/embed.bin --rows 177 178 179 180 181 182 183 184 --std 1.29764e-7 --dist trunc_normal3 --sms 132 148 152 --points 12 --tol-rel 0.05 --tol-abs 0.03
