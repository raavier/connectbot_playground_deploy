-- UC Function para query no Genie Space
-- Execute este script no SQL Editor do Databricks ou em um notebook SQL

CREATE OR REPLACE FUNCTION hs_franquia.gold_connect_bot.query_genie(
  question STRING,
  space_id STRING
)
RETURNS STRING
LANGUAGE PYTHON
AS $$
import requests
import os

def query_genie_space(question: str, space_id: str) -> str:
    """Query Databricks Genie Space"""

    # Databricks workspace URL e token
    workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

    # API endpoint
    url = f"https://{workspace_url}/api/2.0/genie/spaces/{space_id}/start-conversation"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {"content": question}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        # Extrair resposta do Genie
        # (Adaptar baseado na estrutura real do response)
        if "message" in result:
            return result["message"]["content"]
        return str(result)

    except Exception as e:
        return f"Erro ao consultar Genie: {str(e)}"

return query_genie_space(question, space_id)
$$;

-- Teste a função (substitua SEU_SPACE_ID pelo Space ID correto)
SELECT hs_franquia.gold_connect_bot.query_genie(
  'Quantas verificações tivemos em 2024?',
  'SEU_SPACE_ID_AQUI'
);
