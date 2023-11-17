## Important notice

```OrderBookSaver``` is does not work if you use raw ```pams.market``` because ```pams.logs.logger``` does not save logs in sequential (time series) order. Therefore, please replace ```pams/market.py``` to ```market_.py``` [here](https://drive.google.com/file/d/1eWW4kQSAo0VLPQ96b2r8xOAdq1DZ_EUk/view?usp=share_link) or rewrite the ```pams/market.py``` code in your virtual environment as follows.

- l656 in ```._cancel_order``` method

before: ```log.read_and_write(logger=self.logger)```
after: ```log.read_and_write_with_direct_process(logger=self.logger)```

- l727 in ```._execute_orders``` method

before: ```log.read_and_write(logger=self.logger)```
after: ```log.read_and_write_with_direct_process(logger=self.logger)```

- l776 in ```._add_order``` method

before: ```log.read_and_write(logger=self.logger)```
after: ```log.read_and_write_with_direct_process(logger=self.logger)```

- comment out l930~932 in ```._execution``` method