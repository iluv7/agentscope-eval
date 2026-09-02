"""Launch the local evaluation service."""

import argparse

import uvicorn


def main():
    """Run the service on loopback unless explicitly configured otherwise."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8787, type=int)
    args = parser.parse_args()
    uvicorn.run(
        "agentscope_eval.api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
