default:
    @just --list

sync:
    uv sync

fmt:
    uvx ruff format src
    uvx ruff check src --select I --fix

lint:
    uvx ruff format --check src
    uvx ruff check src --select E501,F,I

types:
    uvx pyright@latest

check: lint types

report exp run="same_images":
    open experiments/{{exp}}/runs/{{run}}/report/index.html

publish exp run="same_images":
    rclone sync experiments/{{exp}}/runs/{{run}}/report r2:neural-interp/public/{{exp}}/{{run}}/report --transfers 32
    rclone copyto experiments/{{exp}}/runs/{{run}}/compare.json r2:neural-interp/public/{{exp}}/{{run}}/compare.json
    @echo https://interp.pjay.in/public/{{exp}}/{{run}}/report/index.html

colab:
    lsof -nP -iTCP:7761 -sTCP:LISTEN
