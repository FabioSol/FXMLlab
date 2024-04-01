from Loader import data_dir
import pandas as pd
import os


class Reader:
    def __init__(self, *args, **kwargs):
        if 'filename' in kwargs:
            self.file_name = kwargs['filename']
        elif len(args) == 1:
            self.file_name = args[0]
        else:
            self.file_name = Reader.file_name_from_params(*args, **kwargs)
        self.file_assertion()
        self.data = self.read()

    def file_assertion(self) -> None:
        if self.file_name not in os.listdir(data_dir):
            raise FileNotFoundError(f"the file {self.file_name} is not in {data_dir}")
        if self.file_name.split(".")[-1] != "csv":
            raise NotImplementedError

    @staticmethod
    def file_name_from_params(symbol: str = 'EURUSD', timeframe: str = 'D1', gmt: str = 0) -> str:
        return f"{symbol.upper()}_GMT+{gmt}_NO-DST_{timeframe.upper()}.csv"

    def read(self) -> pd.DataFrame:
        df = pd.read_csv(data_dir + self.file_name)
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df.set_index('datetime', inplace=True)
        df.drop(['Date', 'Time'], axis=1, inplace=True)
        return df

    @staticmethod
    def read_file(*args, **kwargs) -> pd.DataFrame:
        return Reader(*args, **kwargs).data
