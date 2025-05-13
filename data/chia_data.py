import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Đọc dữ liệu từ file CSV gốc
    df = pd.read_csv("data\CreditCardData.csv")

    # Chia dữ liệu
    train_df, test_temp = train_test_split(df, test_size=0.2, random_state=42)
    dev_df, test_df = train_test_split(test_temp, test_size=0.5, random_state=42)

    # Lưu các tập
    train_df.to_csv("data/train.csv", index=False)
    dev_df.to_csv("data/dev.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("Đã chia dữ liệu thành train.csv, dev.csv và test.csv")

if __name__ == "__main__":
    main()
