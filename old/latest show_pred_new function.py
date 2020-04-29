def show_pred_new(epoch, sample_idx=None):
  if sample_idx is None:
    sample_idx = random.randrange(len(x_train))
  x = x_train[sample_idx]
  y = y_train[sample_idx]

  input_labels = support.model_inputs
  output_labels = support.model_outputs
  _scales = [scales[i] for i in support.model_inputs + support.model_outputs]
  colors = ['r', 'g', 'b', 'm', 'c', 'r']

  for idx, (lbl, clr) in enumerate(zip(input_labels, colors)):
    ax[idx].cla()
    if recurrent:
      lst = support.unnorm(np.take(x, axis=1, indices=idx), lbl)
    else:
      lst = support.unnorm(x[idx::len(input_labels)], lbl)
    if lbl == 'delta_desired':
      lst = np.degrees(lst * 17.8)
    if recurrent:
      ax[idx].plot(range(1, len(x) + 1), lst, clr)
    else:
      ax[idx].plot(range(1, (len(x) // len(input_labels)) + 1), lst, clr)
  ax[-1].cla()

  for idx, (lbl, clr) in enumerate(zip(output_labels, colors)):
    if recurrent:
      if not support.one_sample:
        lst = support.unnorm(np.take(y, axis=1, indices=idx), lbl)
      else:
        lst = support.unnorm(y, lbl)
    else:
      lst = support.unnorm(y[idx::len(output_labels)], lbl)
    if lbl == 'delta_desired':
      lst = np.degrees(lst * 17.8)
    if recurrent:
      ax[idx].plot(range(len(x), len(x) + len(y)), lst, clr, label=lbl)
    else:
      ax[idx].plot(range((len(x) // len(input_labels)), len(x) // len(input_labels) + len(y) // len(output_labels)), lst, clr, label=lbl)

  pred = model.predict(np.array([x]))[0]
  for idx, (lbl, clr) in enumerate(zip(output_labels, colors)):
    if recurrent:
      if not support.one_output_feature:
        lst = support.unnorm(np.take(pred, axis=1, indices=idx), lbl)
      else:
        lst = support.unnorm(pred, lbl)
    else:
      lst = support.unnorm(pred[idx::len(output_labels)], lbl)
    if lbl == 'delta_desired':
      lst = np.degrees(lst * 17.8)
    if recurrent:
      ax[idx].plot(range(len(x), len(x) + len(y)), lst, label=lbl + ' pred')
    else:
      ax[idx].plot(range((len(x) // len(input_labels)), len(x) // len(input_labels) + len(y) // len(output_labels)), lst, label=lbl + ' pred')
    ax[idx].legend(loc='upper left')

  fig.show()
  plt.pause(0.01)
  plt.savefig('models/model_imgs/{}'.format(epoch))