(() => {
  'use strict';

  const root = document.querySelector('[data-quiz]');
  const form = document.querySelector('[data-question-form]');
  const state = window.GeoQuizState;
  if (!root || !form || !state) return;

  const timer = root.querySelector('[data-timer]');
  const saveState = root.querySelector('[data-save-state]');
  const sequenceInput = form.querySelector('[data-activity-seq]');
  const activeSecondsInput = form.querySelector('[data-active-seconds]');
  const choices = [...form.querySelectorAll('input[name="selected_codes[]"]')];
  const csrf = root.dataset.csrf || '';
  const saveUrl = form.action;
  const submitUrl = saveUrl.replace(/\/answers$/, '/submit');
  const initialRemaining = Number(root.dataset.remainingSeconds || 1800);
  const timerStarted = performance.now();
  let activeSeconds = 0;
  let saveTimer = 0;
  let timerExpired = false;
  let saving = null;
  let queued = false;
  let lastActivityAt = Date.now();

  const setSaveState = (text, state = '') => {
    saveState.textContent = text;
    saveState.className = `save-state${state ? ` is-${state}` : ''}`;
  };

  const selectedCodes = () => choices.filter((choice) => choice.checked).map((choice) => choice.value);

  const bodyForSave = () => ({
    _csrf: csrf,
    question_code: form.elements.question_code.value,
    selected_codes: selectedCodes(),
    activity_seq: Number(sequenceInput.value),
    active_seconds_delta: state.cappedActiveDelta(activeSeconds),
  });

  const fetchWithRetry = async (body) => {
    const delays = [0, 1000, 2000, 4000, 8000];
    let lastError = new Error('网络连接失败。');
    for (let attempt = 0; attempt < delays.length; attempt += 1) {
      if (delays[attempt] > 0) {
        setSaveState(`重试 ${attempt}/4`, 'error');
        await new Promise((resolve) => window.setTimeout(resolve, delays[attempt]));
      }
      try {
        const response = await fetch(saveUrl, {
          method: 'POST',
          credentials: 'same-origin',
          headers: {
            Accept: 'application/json',
            'Content-Type': 'application/json',
            'X-CSRF-Token': csrf,
          },
          body: JSON.stringify(body),
        });
        if (response.status < 500) return response;
        lastError = new Error(`服务暂时不可用（${response.status}）。`);
      } catch (error) {
        lastError = error instanceof Error ? error : lastError;
      }
    }
    throw lastError;
  };

  const save = async () => {
    if (saving) {
      queued = true;
      return saving;
    }
    const body = bodyForSave();
    const sentSeconds = body.active_seconds_delta;
    setSaveState('保存中', 'saving');
    saving = fetchWithRetry(body)
      .then(async (response) => {
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          if (data.redirect) location.assign(data.redirect);
          throw new Error(data.error || data.message || '答案暂未保存。');
        }
        const responseSequence = data.stale ? Number(data.activity_seq) + 1 : Number(body.activity_seq) + 1;
        sequenceInput.value = String(state.mergeActivitySequence(sequenceInput.value, responseSequence));
        if (!data.stale) {
          activeSeconds = Math.max(0, activeSeconds - sentSeconds);
        }
        activeSecondsInput.value = String(activeSeconds);
        setSaveState('已保存');
        return data;
      })
      .catch((error) => {
        setSaveState('待重试', 'error');
        throw error;
      })
      .finally(() => {
        saving = null;
        if (queued) {
          queued = false;
          window.queueMicrotask(() => save().catch(() => {}));
        }
      });
    return saving;
  };

  const queueSave = () => {
    window.clearTimeout(saveTimer);
    setSaveState('待保存', 'saving');
    saveTimer = window.setTimeout(() => save().catch(() => {}), 420);
  };

  choices.forEach((choice) => choice.addEventListener('change', () => {
    sequenceInput.value = String(state.nextActivitySequence(sequenceInput.value));
    queueSave();
  }));

  form.addEventListener('submit', () => {
    window.clearTimeout(saveTimer);
    activeSecondsInput.value = String(state.cappedActiveDelta(activeSeconds));
  });

  const submitTrigger = form.querySelector('[data-submit-test]');
  const submitDialog = root.querySelector('[data-submit-dialog]');
  const submitDialogConfirm = submitDialog?.querySelector('[data-submit-dialog-confirm]');
  const submitDialogCancelButtons = submitDialog ? [...submitDialog.querySelectorAll('[data-submit-dialog-cancel]')] : [];
  const submitAnswered = submitDialog?.querySelector('[data-submit-answered]');
  const submitUnanswered = submitDialog?.querySelector('[data-submit-unanswered]');

  const closeSubmitDialog = (restoreFocus = true) => {
    if (!submitDialog || submitDialog.hidden) return;
    submitDialog.hidden = true;
    document.body.classList.remove('is-submit-dialog-open');
    submitTrigger?.setAttribute('aria-expanded', 'false');
    if (restoreFocus) submitTrigger?.focus();
  };

  submitTrigger?.addEventListener('click', (event) => {
    if (!submitDialog || !submitDialogConfirm || !submitAnswered || !submitUnanswered) return;
    event.preventDefault();
    const summary = state.submissionSummary(
      root.dataset.answered,
      root.dataset.currentAnswered,
      selectedCodes().length,
    );
    submitAnswered.textContent = String(summary.answered);
    submitUnanswered.textContent = String(summary.unanswered);
    submitDialog.hidden = false;
    document.body.classList.add('is-submit-dialog-open');
    submitTrigger.setAttribute('aria-expanded', 'true');
    const continueButton = submitDialogCancelButtons[submitDialogCancelButtons.length - 1];
    if (continueButton) continueButton.focus();
  });

  submitDialogConfirm?.addEventListener('click', () => {
    closeSubmitDialog(false);
    form.requestSubmit(submitTrigger);
  });
  submitDialogCancelButtons.forEach((button) => button.addEventListener('click', () => closeSubmitDialog()));
  submitDialog?.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      closeSubmitDialog();
      return;
    }
    if (event.key !== 'Tab') return;
    const focusable = [...submitDialog.querySelectorAll('button:not([disabled])')].filter((button) => button.tabIndex >= 0);
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  });

  const updateTimer = () => {
    const elapsed = (performance.now() - timerStarted) / 1000;
    const remaining = Math.max(0, Math.ceil(initialRemaining - elapsed));
    const minutes = Math.floor(remaining / 60);
    const seconds = remaining % 60;
    timer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    timer.dateTime = `PT${minutes}M${seconds}S`;
    timer.classList.toggle('is-urgent', remaining <= 300);
    if (remaining === 0 && !timerExpired) {
      timerExpired = true;
      const submitter = document.createElement('button');
      submitter.type = 'submit';
      submitter.formAction = submitUrl;
      submitter.hidden = true;
      form.append(submitter);
      form.requestSubmit(submitter);
    }
  };

  updateTimer();
  window.setInterval(updateTimer, 1000);
  window.setInterval(() => {
    if (state.shouldCountActiveSecond(document.visibilityState, document.hasFocus(), lastActivityAt, Date.now())) {
      activeSeconds += 1;
      activeSecondsInput.value = String(state.cappedActiveDelta(activeSeconds));
      if (activeSeconds >= 30) save().catch(() => {});
    }
  }, 1000);
  const markActivity = () => { lastActivityAt = Date.now(); };
  document.addEventListener('pointerdown', markActivity, { passive: true });
  document.addEventListener('keydown', markActivity);
  document.addEventListener('keydown', (event) => {
    if (submitDialog && !submitDialog.hidden) return;
    if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return;
    const editable = event.target instanceof HTMLInputElement && !choices.includes(event.target);
    if (editable) return;
    const number = Number(event.key);
    if (number >= 1 && number <= choices.length) {
      event.preventDefault();
      choices[number - 1].click();
      choices[number - 1].focus();
      return;
    }
    if (event.key === 'ArrowLeft') {
      const previous = form.querySelector('button[name="navigate_to"]:first-of-type');
      if (previous?.value < root.dataset.position) {
        event.preventDefault();
        form.requestSubmit(previous);
      }
    }
    if (event.key === 'ArrowRight') {
      const next = [...form.querySelectorAll('button[name="navigate_to"]')].find((button) => Number(button.value) > Number(root.dataset.position));
      if (next) {
        event.preventDefault();
        form.requestSubmit(next);
      }
    }
  });

  const drawer = root.querySelector('[data-mobile-nav]');
  const toggle = root.querySelector('[data-nav-toggle]');
  const close = root.querySelector('[data-nav-close]');
  const closeDrawer = () => {
    drawer.hidden = true;
    toggle.setAttribute('aria-expanded', 'false');
    document.body.classList.remove('is-nav-open');
    toggle.focus();
  };
  toggle?.addEventListener('click', () => {
    drawer.hidden = false;
    toggle.setAttribute('aria-expanded', 'true');
    document.body.classList.add('is-nav-open');
    close?.focus();
  });
  close?.addEventListener('click', closeDrawer);
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !drawer.hidden) closeDrawer();
  });

  window.addEventListener('pagehide', () => {
    if (!navigator.sendBeacon || activeSeconds === 0) return;
    const params = new URLSearchParams();
    const payload = bodyForSave();
    params.set('_csrf', payload._csrf);
    params.set('question_code', payload.question_code);
    params.set('activity_seq', String(payload.activity_seq));
    params.set('active_seconds_delta', String(payload.active_seconds_delta));
    payload.selected_codes.forEach((code) => params.append('selected_codes[]', code));
    navigator.sendBeacon(saveUrl, new Blob([params.toString()], { type: 'application/x-www-form-urlencoded;charset=UTF-8' }));
  });
})();
