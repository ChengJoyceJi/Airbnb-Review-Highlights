from data import Data

def main():
    data = Data()
    data.parse_input_data()
    print(len(data.list_id_to_reviews))
    
if __name__ == '__main__':
    main()