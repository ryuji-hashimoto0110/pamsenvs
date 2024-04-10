## Important note

The loggers in this directory do not work if you use raw ```pams.market``` because ```pams.logs.logger``` does not save logs in sequential (time series) order and does not write log at log.time. Therefore, please replace ```pams/market.py``` to ```market_.py``` [here](https://drive.google.com/file/d/1iDoWINiyDzXa0q4upMOUCkuomT2jJvWu/view?usp=share_link) or replace all ```.read_and_write()``` in ```pams/market.py``` to ```.read_and_write_with_direct_process()``` in your virtual environment. Also, comment out the following code in ```_execution()```.

```python
if self.logger is not None:
    self.logger.bulk_write(logs=cast(List[Log], logs))
```

## Remaining bugs or malfunctions

- coloring submitted orders by specific agents by designating ```specific_agent_color_dic``` sometimes fails.
- disturbance in a video image sometimes occurr in specified ```cv2.VideoWriter``` parameters. Please refer to [```examples/market_impact/main.py```](https://github.com/ryuji-hashimoto0110/pams_environments/blob/main/examples/market_impact/main.py) for successfully implemented example.


