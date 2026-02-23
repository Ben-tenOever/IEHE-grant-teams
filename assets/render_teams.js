function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k === "text") node.textContent = v;
    else if (v !== undefined && v !== null) node.setAttribute(k, v);
  }
  for (const ch of children) node.appendChild(ch);
  return node;
}

function safe(x) {
  return (x ?? "").toString();
}

function parseTeam(teamMembersStr) {
  return safe(teamMembersStr)
    .split(";")
    .map(s => s.trim())
    .filter(Boolean);
}

function parseAims(aimsRaw) {
  const t = safe(aimsRaw).trim();
  if (!t) return [];
  if (t.includes(" | ")) return t.split(" | ").map(s => s.trim()).filter(Boolean);
  if (t.includes("\n")) return t.split(/\n+/).map(s => s.trim()).filter(Boolean);
  return [t];
}

function fmtMoney(x) {
  if (x === undefined || x === null || x === "") return "";
  const n = Number(x);
  if (!Number.isFinite(n)) return "";
  try {
    return new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(n);
  } catch {
    return "$" + Math.round(n).toString();
  }
}

async function loadTeams(jsonPath, embeddedId) {
  // When opening pages directly from disk (file://), browsers often block fetch() of local files.
  // In that case, we fall back to embedded JSON data, if present.
  const tryEmbedded = () => {
    if (!embeddedId) return null;
    const el = document.getElementById(embeddedId);
    if (!el) return null;
    const txt = (el.textContent || "").trim();
    if (!txt) return null;
    try {
      return JSON.parse(txt);
    } catch (e) {
      console.warn("Failed to parse embedded data", e);
      return null;
    }
  };

  const embedded = tryEmbedded();
  if (window.location && window.location.protocol === "file:" && embedded) return embedded;

  if (jsonPath) {
    try {
      const resp = await fetch(jsonPath);
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      return await resp.json();
    } catch (e) {
      // If fetch fails (common on file://), fall back to embedded data if available.
      if (embedded) return embedded;
      throw new Error("Failed to load data from " + jsonPath + ": " + (e && e.message ? e.message : e));
    }
  }

  if (embedded) return embedded;
  throw new Error("No embedded data and no data path provided");
}

function renderMeta(rec) {
  const rows = [];

  const addText = (label, value) => {
    const v = safe(value).trim();
    if (!v) return;
    rows.push(el("div", { class: "meta-row" }, [
      el("div", { class: "meta-label", text: label }),
      el("div", { class: "meta-value", text: v })
    ]));
  };

  const addLink = (label, text, href) => {
    const t = safe(text).trim();
    const h = safe(href).trim();
    if (!t) return;
    if (!h) { addText(label, t); return; }
    rows.push(el("div", { class: "meta-row" }, [
      el("div", { class: "meta-label", text: label }),
      el("div", { class: "meta-value" }, [
        el("a", { href: h, target: "_blank", rel: "noopener noreferrer" }, [document.createTextNode(t)])
      ])
    ]));
  };

  addText("Agency", rec.agency_name);
  addLink("Announcement ID", rec.opportunity_number, rec.opportunity_url);
  addText("Status", rec.opp_status);
  addText("Open date", rec.open_date);
  addText("Close date", rec.close_date);

  if (!rows.length) return null;

  return el("div", { class: "section" }, [
    el("div", { class: "section-label", text: "Opportunity details" }),
    el("div", { class: "meta" }, rows)
  ]);
}

function renderCard(rec, opts = {}) {
  const title = safe(rec.opportunity_title);
  const agency = safe(rec.agency_name);
  const oppNum = safe(rec.opportunity_number);
  const closeDate = safe(rec.close_date);
  const url = safe(rec.opportunity_url);

  const subtitleBits = [];
  if (agency) subtitleBits.push(document.createTextNode(agency));
  if (oppNum) {
    subtitleBits.push(document.createTextNode(" • "));
    subtitleBits.push(url ? el("a", { href: url, target: "_blank", rel: "noopener noreferrer" }, [document.createTextNode(oppNum)]) : document.createTextNode(oppNum));
  }
  if (closeDate) subtitleBits.push(document.createTextNode(" • closes " + closeDate));

  const header = el("div", { class: "card-header" }, [
    el("div", { class: "card-title" }, [
      el("a", { href: url || "#", target: "_blank", rel: "noopener noreferrer" }, [document.createTextNode(title || "Untitled")])
    ]),
    el("div", { class: "card-subtitle" }, subtitleBits)
  ]);

  const team = parseTeam(rec.team_members);
  const teamLine = el("div", { class: "pill-row" }, team.map(n => el("span", { class: "pill" }, [document.createTextNode(n)])));

  const proposed = el("div", { class: "section" }, [
    el("div", { class: "section-label", text: "Suggested title" }),
    el("div", { class: "section-text", text: safe(rec.proposed_title) })
  ]);

  const rationale = el("div", { class: "section" }, [
    el("div", { class: "section-label", text: "Rationale" }),
    el("div", { class: "section-text", text: safe(rec.rationale) })
  ]);

  const aims = parseAims(rec.aims_or_projects);
  const aimsList = el("ol", { class: "aims" }, aims.map(a => el("li", {}, [document.createTextNode(a)])));
  const aimsWrap = el("div", { class: "section" }, [
    el("div", {
      class: "section-label",
      text: rec.team_size_category === "large" ? "Proposed projects" : "Proposed aims"
    }),
    aimsList
  ]);

  const meta = renderMeta(rec);

  const body = el("div", { class: "details-body" }, [
    proposed,
    rationale,
    aimsWrap,
    ...(meta ? [meta] : [])
  ]);

  if (opts.alwaysExpanded) {
    return el("div", { class: "card" }, [header, teamLine, body]);
  }

  const details = el("details", { class: "details" }, [
    el("summary", { class: "summary", text: "Show rationale and aims" }),
    body
  ]);
  if (opts.openByDefault) details.open = true;

  return el("div", { class: "card" }, [header, teamLine, details]);
}

function applyFilter(recs, query) {
  const q = safe(query).trim().toLowerCase();
  if (!q) return recs;
  return recs.filter(r => {
    const hay = [
      r.opportunity_title,
      r.agency_name,
      r.opportunity_number,
      r.opp_status,
      r.team_members,
      r.proposed_title,
      r.rationale,
      r.aims_or_projects
    ].map(safe).join(" ").toLowerCase();
    return hay.includes(q);
  });
}

function computeTopIds(recs, topN) {
  const ids = recs
    .slice(0, Math.max(0, topN))
    .map(r => safe(r.opportunity_id))
    .filter(Boolean);
  return new Set(ids);
}

async function main() {
  const container = document.getElementById("cards");
  const search = document.getElementById("search");
  const count = document.getElementById("count");

  const jsonPath = container.getAttribute("data-json");
  const embeddedId = container.getAttribute("data-embedded");
  const mode = safe(container.getAttribute("data-mode")).trim();
  const topN = Number(container.getAttribute("data-topn") || 0);
  const openByDefault = safe(container.getAttribute("data-open")) === "1";
  const alwaysExpanded = safe(container.getAttribute("data-expanded")) === "1";

  const recsAll = await loadTeams(jsonPath, embeddedId);
  const topIds = computeTopIds(recsAll, topN);

  let base = recsAll;
  if (mode === "top") {
    base = recsAll.slice(0, topN);
  } else if (mode === "others") {
    base = recsAll.filter(r => !topIds.has(safe(r.opportunity_id)));
  }

  function rerender() {
    const filtered = applyFilter(base, search ? search.value : "");
    container.innerHTML = "";
    for (const r of filtered) container.appendChild(renderCard(r, { openByDefault, alwaysExpanded }));
    if (count) count.textContent = `${filtered.length} opportunities`;
  }

  if (search) search.addEventListener("input", rerender);
  rerender();
}

window.addEventListener("DOMContentLoaded", () => {
  main().catch(err => {
    const container = document.getElementById("cards");
    if (container) container.innerHTML = "<div class='error'>Error loading data: " + safe(err.message) + "</div>";
  });
});
