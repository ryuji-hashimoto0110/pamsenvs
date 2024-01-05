## Create artificial OLHCV data.

Create the following directory structure. via running ```create_artificial_data.sh```.

```
$chmod 755 create_artificial_datas.sh
$./create_artificial_data.sh
```

```
pamsenvs
    |- datas
        |- artificial_datas
        |   |- fcn_wo_cara
        |   |   |- 0.csv
        |   |   |- 1.csv
        |   |   |- ...
        |   |   |- 999.csv
        |   |- fcn
        |       |- 0.csv
        |       |- 1.csv
        |       |- ...
        |   |   |- 999.csv
        |- real_datas
            |- daily
            |   |- aapl
            |   |   |- AAPL_2010-2019.csv
            |   |- sp500
            |       |- SP500_2010-2019.csv
            |- intraday
                |- aapl
                |   |- AAPL_2010-2019.csv
                |- sp500
                    |- SP500_2010-2019.csv
```