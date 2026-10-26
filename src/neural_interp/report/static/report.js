function tab(group, index) {
  document.querySelectorAll('[data-g="' + group + '"]').forEach(
    (element) => element.classList.toggle('on', element.dataset.n === index)
  );
  window.dispatchEvent(new Event('resize'));
}

function upclose() {
  const picker = document.querySelector('.picker');
  const trees = Number(picker.dataset.trees);
  const on = [...document.querySelectorAll('.pick.on')].map((button) => Number(button.dataset.t));
  const visible = [];
  for (let t = 0; t < trees; t++) {
    const shown = on.includes(t);
    visible.push(shown, shown, shown);
  }
  Plotly.restyle('upclose', { visible: visible });
  Plotly.relayout('upclose', { 'xaxis.autorange': true, 'yaxis.autorange': true });
}

function pick(button) {
  button.classList.toggle('on');
  upclose();
}

function pickAll(on) {
  document.querySelectorAll('.pick').forEach((button) => button.classList.toggle('on', on));
  upclose();
}
