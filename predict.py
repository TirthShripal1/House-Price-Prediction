import argparse
from src.predict import predict_price

def main():
    parser = argparse.ArgumentParser(description="Predict a house price.")
    parser.add_argument("--area", type=float, required=True, help="Area in square feet")
    parser.add_argument("--bedrooms", type=int, required=True)
    parser.add_argument("--bathrooms", type=int, required=True)
    parser.add_argument("--stories", type=int, required=True)
    parser.add_argument("--parking", type=int, required=True)
    parser.add_argument("--location", choices=["urban", "suburban", "rural"], required=True)
    parser.add_argument("--condition", choices=["excellent", "good", "average", "needs_renovation"], required=True)
    args = parser.parse_args()
    sample = {
        "area_sqft": args.area, "bedrooms": args.bedrooms,
        "bathrooms": args.bathrooms, "stories": args.stories,
        "parking": args.parking, "location": args.location,
        "condition": args.condition,
    }
    result = predict_price(sample)
    print(f"Estimated demonstration price: INR {result:,.0f}")

if __name__ == "__main__":
    main()
