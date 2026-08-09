(() => {
  'use strict';

  const nameForm = document.querySelector('[data-validate-name]');
  if (nameForm) {
    nameForm.addEventListener('submit', (event) => {
      const input = nameForm.querySelector('input[name="name"]');
      const name = input.value.trim().replace(/\s+/g, ' ');
      let error = nameForm.querySelector('[data-client-name-error]');
      if (!error) {
        error = document.createElement('p');
        error.className = 'field-error';
        error.dataset.clientNameError = '';
        error.setAttribute('role', 'alert');
        nameForm.append(error);
      }
      if ([...name].length < 2 || [...name].length > 40) {
        event.preventDefault();
        error.textContent = '请输入 2 至 40 个字符的姓名。';
        input.setAttribute('aria-invalid', 'true');
        input.focus();
        return;
      }
      input.value = name;
      input.removeAttribute('aria-invalid');
      error.remove();
    });
  }

  document.querySelectorAll('.delete-disclosure form').forEach((form) => {
    form.addEventListener('submit', (event) => {
      const expected = form.querySelector('label')?.textContent.match(/「(.+)」/)?.[1] || '';
      const actual = form.querySelector('input[name="confirmation_name"]')?.value.trim() || '';
      if (actual !== expected) {
        event.preventDefault();
        const input = form.querySelector('input[name="confirmation_name"]');
        input?.setAttribute('aria-invalid', 'true');
        input?.focus();
      }
    });
  });
})();
