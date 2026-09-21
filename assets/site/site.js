"use strict";

// All publication content and the default method are present without JavaScript.
document.documentElement.classList.add("js");
// Match the RLT template's fixed navigation and full-screen mobile menu.
const navbar = document.querySelector(".navbar");
const burger = document.querySelector(".navbar-burger");
const menu = document.getElementById("navbarMenu");
burger.hidden = false;
function setMenu(open) {
  burger.classList.toggle("is-active", open);
  menu.classList.toggle("is-active", open);
  navbar.classList.toggle("nav-expanded", open);
  document.body.classList.toggle("nav-open", open);
  burger.setAttribute("aria-expanded", String(open));
  burger.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
  menu.inert = mobile.matches && !open;
}
const mobile = window.matchMedia("(max-width: 820px)");
burger.addEventListener("click", () => setMenu(burger.getAttribute("aria-expanded") !== "true"));
menu.querySelectorAll("a").forEach(link => link.addEventListener("click", () => setMenu(false)));
document.addEventListener("keydown", event => {
  if (burger.getAttribute("aria-expanded") !== "true") return;
  if (event.key === "Escape") {
    setMenu(false);
    burger.focus();
  }
  // Keep keyboard focus inside the open mobile navigation.
  if (event.key === "Tab" && mobile.matches) {
    const first = navbar.querySelector(".navbar-brand a");
    const last = menu.querySelector(".navbar-end a:last-child");
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }
});
mobile.addEventListener("change", () => setMenu(false));
setMenu(false);
if ("IntersectionObserver" in window) {
  new IntersectionObserver(([entry]) => navbar.classList.toggle("is-scrolled", !entry.isIntersecting),
    {rootMargin: "-69px 0px 0px 0px"}).observe(document.querySelector(".hero"));
} else {
  navbar.classList.add("is-scrolled");
}

const estimators = {
  mc: {
    name: "MC-KL",
    records: "M independent token draws per prefix, with replacement.",
    tokenGradient: "−∑_u h_u [∇ log p(a_u) − (1/M) ∑_j ∇ log p(v_u,j)]",
    sequenceGradient: "−(1/M) ∑_j D_−j ∑_u [∇ log p(a_u) − ∇ log p(v_u,j)]",
    tokenProperty: "Independent auxiliary draws recover the full-KL token gradient in expectation, including M = 1.",
    sequenceProperty: "D_j = R − β∑_u [ℓ_u + log q(v_u,j) − log p(v_u,j)]; D_−j averages D_l for l ≠ j. Leave-one-out feedback avoids correlated residual/score estimates and recovers the full-KL sequence gradient in expectation.",
    api: "klpo_sequence_mc_loss",
    flags: ""
  },
  topk: {
    name: "TopK-KL",
    records: "The sampler’s K highest-probability tokens with their original probabilities; all other tokens form one tail bucket. No head renormalization. Launcher default K = 128.",
    tokenGradient: "−∑_u h_u [∇ log p(a_u) − ∑_{v∈H_K} (q_v − ρ_u p_v) ∇ log p_v]",
    sequenceGradient: "−D ∑_u [∇ log p(a_u) + ∇ k_u]",
    tokenProperty: "Top-K Aggregated KL keeps a head and an aggregate tail. Here ρ_u = q_tail / p_tail at prefix u; this formula assumes inactive tail floors. Finite K approximates the full-KL score correction. The implementation uses a floored-ratio approximation when needed.",
    sequenceProperty: "k_u = ∑_{v∈H_K} q_v log(q_v/p_v) + q_tail log(q_tail/p_tail). The displayed expression assumes inactive tail floors. The implementation differentiates the stabilized KL scalar, including the trainer tail. Finite K need not preserve the full-KL population equivalence.",
    api: "klpo_sequence_topk_loss",
    flags: " --kl-estimator topk --top-k 2"
  },
  binary: {
    name: "Binary KL",
    records: "Only the rollout action’s sampler and trainer log-probabilities; the complement forms the second bucket.",
    tokenGradient: "−∑_u h_u ω_u ∇ log p(a_u)",
    sequenceGradient: "−D ∑_u ω_u ∇ log p(a_u)",
    tokenProperty: "ω_u = (1 − q(a_u)) / (1 − p(a_u)). This binary score correction is an approximation and does not generally center the full sampler score exactly.",
    sequenceProperty: "ω_u = (1 − q(a_u)) / (1 − p(a_u)); k_u is the binary KL between action and complement. This is the sample gradient of the binary squared-residual objective, which approximates Full KL.",
    api: "klpo_sequence_loss",
    flags: " --kl-estimator binary"
  },
  full: {
    name: "Full KL",
    records: "The sampler’s full conditional distribution and differentiable trainer scores over the entire vocabulary at every policy prefix.",
    tokenGradient: "−∑_u h_u [∇ log p(a_u) − ∑_v q_v ∇ log p_v]",
    sequenceGradient: "−D ∑_u [∇ log p(a_u) + ∇ k_u]",
    tokenProperty: "The exact sampler-conditioned score mean. It needs full vocabulary records rather than MC draws or an aggregated head/tail partition.",
    sequenceProperty: "k_u = ∑_v q_v log(q_v/p_v) is full conditional KL. This is the sample gradient of D²/(2β). Its population gradient agrees with token regression under the report’s stated assumptions.",
    api: "klpo_sequence_full_loss",
    flags: " --kl-estimator full"
  }
};
const controls = document.getElementById("method-controls");
function updateMethod() {
  const values = new FormData(controls);
  const route = values.get("route");
  const estimator = values.get("estimator");
  const selected = estimators[estimator];
  const token = route === "token";
  const isDefault = token && estimator === "mc";
  document.getElementById("method-title").textContent = `KLPO ${route} regression + ${selected.name}`;
  document.getElementById("method-badge").textContent = isDefault ? "Default" : "Alternative";
  document.getElementById("method-feedback").textContent = token
    ? "Each token uses h_u = R − βℓ_u, where ℓ_u = log p(a_u) − log q(a_u)."
    : estimator === "mc"
      ? "Each auxiliary column j uses a leave-one-out trajectory residual D_−j, shared across the response’s tokens."
      : "A single trajectory residual D = R − β∑_u (ℓ_u + k_u) weights every token; ℓ_u = log p(a_u) − log q(a_u).";
  document.getElementById("method-records").textContent = selected.records + (estimator === "mc" ? ` M ≥ ${token ? 1 : 2}; launcher default M = 128.` : "");
  document.getElementById("method-gradient").textContent = selected[token ? "tokenGradient" : "sequenceGradient"];
  document.getElementById("method-property").textContent = selected[token ? "tokenProperty" : "sequenceProperty"];
  document.getElementById("method-api").textContent = token
    ? `klpo_token_loss(..., kl_estimator="${estimator}")`
    : `${selected.api}(...)`;
  document.getElementById("method-command").textContent = "python examples/train_toy.py" + (token ? "" : " --route sequence") + selected.flags;
}
controls.addEventListener("change", updateMethod);
updateMethod();

const copyButton = document.getElementById("copy-citation");
copyButton.hidden = false;
copyButton.addEventListener("click", async () => {
  const status = document.getElementById("copy-status");
  try {
    await navigator.clipboard.writeText(document.getElementById("bibtex").textContent.trim() + "\n");
    status.textContent = "BibTeX copied.";
  } catch {
    status.textContent = "Select the citation to copy it, or download the .bib file.";
  }
});
