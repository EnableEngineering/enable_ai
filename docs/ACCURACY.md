# Getting Better / More Accurate Results

This guide summarizes how to improve query accuracy and handle edge cases (e.g. user/role filtering, low stock, technicians).

---

## 1. **User / role filtering (“show customer type users”, “which technicians”)**

### What the package does

- **Semantic injection**: When the query mentions a role word (technician, customer, client, admin, manager) and the resolved resource is `users`, the pipeline injects a `role` filter (e.g. `role=Client`, `role=Technician`) if the LLM didn’t already set one. This is done in both the workflow and the orchestrator.
- **Canonical values**: If your API expects different labels (e.g. `Customer` instead of `Client`), configure **resource_hints** so the parser/matcher can map to the exact API value.

### What you should configure

1. **`resource_hints.users` in config**  
   Define the `role` field with allowed values and synonyms so the matcher can validate and map natural language to API values:

   ```json
   "users": {
     "__resource_synonyms__": ["user", "users", "technician", "technicians", "customer", "customers", "staff"],
     "role": {
       "values": ["Admin", "Technician", "Client", "Manager"],
       "synonyms": {
         "technician": "Technician", "tech": "Technician",
         "customer": "Client", "client": "Client", "customer type": "Client",
         "admin": "Admin", "administrator": "Admin",
         "manager": "Manager"
       }
     }
   }
   ```

2. **API schema for `/users/` (or equivalent)**  
   Ensure the list endpoint’s **query parameters** include the filter your API actually supports, e.g.:

   - `role` if the API uses `?role=Client`
   - `role__name` if the API uses Django-style `?role__name=Client`

   If the schema doesn’t list this param, the package may still send it (and add a “filter may not be available” warning). Listing it avoids the warning and makes behavior explicit.

3. **If the API doesn’t support role filtering**  
   The package will still send the filter; the API may ignore it and return all users. In that case you’ll see a **filter warning** in the response. To get “best possible” results you’d need the backend to support a role (or role__name) query param; the package can’t filter server-side without API support.

---

## 2. **Low stock / consumables**

- Semantic injection adds `stock_level=low` when the query contains phrases like “low in stock”, “low stock”, “need to refill” and the resource has a `stock_level` hint.
- In config, define `resource_hints` for the consumables/inventory resource with `stock_level.values` and `stock_level.synonyms` so “low” maps to the exact API value (e.g. `low`).

---

## 3. **Best practices for accuracy (summary)**

| Area | Recommendation |
|------|-----------------|
| **resource_hints** | For every filterable field (role, status, stock_level, etc.), add `values` and `synonyms` so the LLM output is validated and mapped to API values. |
| **Schema query params** | List all supported query params (e.g. `role`, `role__name`, `status__name`) on the list endpoints so the matcher sends the right keys and doesn’t warn. |
| **query_examples** | Use `query_examples` (as a sibling to `resource_hints`, not inside it) for critical phrases so the rule-based classifier or future improvements can align with your API. |
| **user_context** | Pass `user_context` (e.g. `user_id`, `role`) when calling `process()` so “me”, “my”, “assigned to me” resolve correctly. |
| **Filter warnings** | Read `response["filter_warnings"]` and `response["summary"]`; if a filter “may not be available”, the API likely doesn’t support that param — fix schema or backend. |

---

## 4. **Other improvements (package-side)**

- **MAX_TOKENS_DETAILED** was increased (e.g. to 4000) so table summaries for many rows aren’t truncated.
- **TypedDict** and **resource_hints** iteration are hardened so non-dict hint values (e.g. a list) don’t cause crashes.
- **Semantic role injection** is **config-driven**: when `resource_hints.users.role.synonyms` is defined, the package uses it for phrase → value mapping; otherwise it falls back to `constants.ROLE_QUERY_TO_VALUE`. You can define any role names and phrases in config. Order of phrases is tuned so “customer” and “admin” don’t clash.

---

## 5. **What else to handle better**

| Area | What to do |
|------|------------|
| **"Observations for my last report"** | Needs a **multi-step** plan: (1) get my last report, (2) get observations for that report. Ensure your schema exposes the child resource (e.g. observations) and the planner can generate two steps; add relationship or FK hints if needed. |
| **Filter param name (role vs role__name)** | We inject filter key `role`; the matcher maps to endpoint query params (e.g. `role__name`). If your API uses a different param, list it in the endpoint's `parameters.query` in the schema. |
| **API doesn't support a filter** | We send the filter and add a warning. Consider showing a one-line note in your app when `response["filter_warnings"]` is non-empty. |
| **Parser examples** | Parser is instructed to use only resources from the schema. Ensure your schema and resource_hints are loaded so the LLM has the right options. |
| **Rule-based classifier** | The classifier does not extract filters (e.g. role)—only intent, resource, and simple entities. Semantic injection runs afterward in workflow and matcher, so role/stock_level are still applied when the classifier skips the LLM. |
| **Config validation** | If a list (e.g. `query_examples`) is placed inside `resource_hints` by mistake, we skip that key. Consider validating config at load time and logging a warning so deployers move `query_examples` to the correct place. |
