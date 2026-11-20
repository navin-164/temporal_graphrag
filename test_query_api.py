from fastapi.testclient import TestClient
from api.main import app


def main() -> None:
    with TestClient(app) as client:
        resp = client.post(
            "/query",
            json={
                "query": "How has retrieval-augmented generation evolved over time in scientific documents?"
            },
        )
        print("Status:", resp.status_code)
        data = resp.json()
        print("Verified:", data.get("verified"))
        print("Confidence:", data.get("confidence_score"))
        print("Sources:", len(data.get("sources") or []))
        print("Summary:\n", data.get("summary"))


if __name__ == "__main__":
    main()