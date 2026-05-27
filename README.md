State-Space Models (SSMs) have recently shown promising results in time-series modeling, which
often requires complex dependencies over long sequences. SSMs offer an alternative to sequence
models such as RNNs, CNNs, or Transformers. Prior work has investigated SSMs as part of deep
learning layers for more efficient time-series modeling. However, SSMs have limitations in forecasting,
i.e., making future predictions of a sequence given its history. Three forecasting approaches are
studied: zero-masking, closed-loop forecasting, and a SpaceTime-inspired encoder–decoder setup,
using both real-valued and complex-valued state matrices that control the evolution of the hidden
state.

The results show that complex-valued state matrices are critical for forecasting oscillatory signals,
substantially outperforming real-valued matrices on synthetic data. Zero-masking is effective on
low-complexity data, but becomes less robust than the autoregressive models when the signals consist
of multiple frequencies where it requires more training than the SpaceTime-inspired model. On the
univariate ETT-small datasets, the proposed models are competitive and outperform S4 for forecast
horizon 96, but do not consistently outperform SpaceTime. These findings clarify the trade-off
between expressivity, training duration, and that a complex diagonal parameterization provides a
strong framework for forecasting.
