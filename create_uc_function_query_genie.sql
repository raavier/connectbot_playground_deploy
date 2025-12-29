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
import json

def query_genie_space(question: str, space_id: str) -> str:
    """Query Databricks Genie Space"""

    # Get workspace URL and token from environment variables
    # These are automatically set in Databricks UC Functions
    workspace_url = os.environ.get("DATABRICKS_HOST", "").replace("https://", "")
    token = os.environ.get("DATABRICKS_TOKEN", "")

    if not workspace_url or not token:
        return "Error: Databricks credentials not available in environment"

    # API endpoint
    url = f"https://{workspace_url}/api/2.0/genie/spaces/{space_id}/start-conversation"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {"content": question}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        # Extrair resposta do Genie
        # A estrutura pode variar, então vamos retornar o que conseguirmos
        if isinstance(result, dict):
            # Tentar pegar o conteúdo da mensagem
            if "message" in result and isinstance(result["message"], dict):
                if "content" in result["message"]:
                    return result["message"]["content"]

            # Tentar pegar attachments (query results)
            if "attachments" in result:
                attachments = result["attachments"]
                if attachments and len(attachments) > 0:
                    first_attachment = attachments[0]
                    if "text" in first_attachment:
                        return first_attachment["text"]["content"]
                    if "query" in first_attachment:
                        query_result = first_attachment["query"]
                        if "result" in query_result:
                            return json.dumps(query_result["result"])

            # Se não encontrou, retornar o JSON completo
            return json.dumps(result, ensure_ascii=False, indent=2)

        return str(result)

    except requests.exceptions.Timeout:
        return "Error: Genie query timed out after 60 seconds"
    except requests.exceptions.RequestException as e:
        return f"Error calling Genie API: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

return query_genie_space(question, space_id)
$$;

-- Teste a função (substitua SEU_SPACE_ID pelo Space ID correto)
SELECT hs_franquia.gold_connect_bot.query_genie(
  'Quantas verificações tivemos em 2024?',
  'SEU_SPACE_ID_AQUI'
);
