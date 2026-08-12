"""Offline PyTorch LSTM training scaffold for ISYARA.

This script intentionally does not run during Django/FastAPI startup.
"""


def main():
    print("Training scaffold.")
    print("Expected output: services/assessment_service/assessment_service/models/sign_classifier.pt")
    print("Use signer-independent splits when metadata is available.")


if __name__ == "__main__":
    main()
