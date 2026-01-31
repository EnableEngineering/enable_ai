# Backend simulator (DEV config)

Used by `environment.py` when `ACTIVE_ENVIRONMENT = 'DEV'`. Copy `env.example` to `.env` and set `OPENAI_API_KEY` (and optionally `API_EMAIL` / `API_PASSWORD` for JWT).

- **config.json** – API data source, schema path, JWT config.
- **schemas/api_schema.json** – Minimal API schema; replace with your OpenAPI-derived schema (`enable-schema generate --input openapi.json --base-url ... --output schemas/api_schema.json`).

To use: set your project’s config path to this folder, or run with `cwd` in this directory so config and schemas resolve.
