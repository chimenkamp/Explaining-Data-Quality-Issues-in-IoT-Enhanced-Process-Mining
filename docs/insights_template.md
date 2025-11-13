# Backus–Naur Form (BNF)
```html
<ProcessMiningResult> ::= 
    <Header>
    <ContextualExplanation>
    <NarrativeLayer>
    <ActionMapping>
    <ConfidenceAndAffect>
    <References>

;---------------------------------------------------------------
; HEADER — clear purpose & goal framing
; Tentina et al. (2025, F9) show consumers need “Clear Goal and Business Problem”
; before analysis to understand what insight means.
; Ammann et al. (2025) stress role-dependent framing.
;---------------------------------------------------------------
<Header> ::= 
    "Insight ID: " <ID> "\n"
    "Process Area: " <ProcessArea> "\n"
    "Business Goal: " <GoalStatement> "\n"
    "Analysis Type: " <AnalysisMode> "\n"
    "Relevance: " <MetricSummary> "\n"

<AnalysisMode> ::= "Directed" | "Exploratory"   ; from Tentina 2025, Sec. V-B
<GoalStatement> ::= <sentence>
<MetricSummary> ::= <percentage> " deviation in " <metric> " from baseline"

;---------------------------------------------------------------
; CONTEXTUAL EXPLANATION — transparency of what and how
; Responds to Tentina F1-F3: “General Concepts”, “Pipeline Transparency”, 
; “Reference Examples”; and R2 recommendation on “explainable outputs”.
;---------------------------------------------------------------
<ContextualExplanation> ::= 
    "Observed Behavior: " <BehaviorDescription> "\n"
    "Underlying Data Scope: " <DataScope> "\n"
    "Applied Filters / Pipeline Notes: " <PipelineNote> "\n"
    "Context: " <BusinessContext> "\n"

<DataScope> ::= "Event log: " <LogID> ", timeframe: " <timeWindow>
<PipelineNote> ::= <sentence> ; explain preprocessing or aggregation
<BusinessContext> ::= <sentence>

;---------------------------------------------------------------
; NARRATIVE LAYER — sense-making and role-specific storytelling
; Derived from Ammann et al. (2025) user archetypes and 
; Tentina F10 (Focus on Requirements & Validation) + F11 (Business Knowledge).
;---------------------------------------------------------------
<NarrativeLayer> ::= 
    <ExecutiveSummary> 
    [<AnalystDetail>]

<ExecutiveSummary> ::= 
    "Summary: " <sentence> "\n"
    "Impact: " <impactMetric> " (" <businessImpact> ")" "\n"
    "Key Takeaway: " <sentence> "\n"

<AnalystDetail> ::= 
    "Analyst Notes: " <DeepDiveText> "\n"
    "Linked KPIs: " <metricList> "\n"
    "Supporting Examples: " <ExampleRefs> "\n"

<ExampleRefs> ::= "Reference dashboards/examples: " <string> ; Tentina F3

;---------------------------------------------------------------
; ACTION MAPPING — link to process improvement
; Confirms Stein Dani et al. (2024) on bridging “insight → action” gap,
; plus Tentina F10 (user validation and ownership).
;---------------------------------------------------------------
<ActionMapping> ::= 
    "Recommended Actions:" "\n"
    { <ActionItem> "\n" }

<ActionItem> ::= 
    "- " <ActionVerb> " " <ActionObject>
      " (Responsible: " <Actor> ", Priority: " <PriorityLevel> ")"
      [". Expected Outcome: " <Outcome> "]"

<ActionVerb> ::= "Review" | "Investigate" | "Automate" | "Escalate" | "Reassign"
<ActionObject> ::= <processStep> | <systemRule> | <resource>
<Actor> ::= <role> | "Process Owner" | "Analyst" | "IT Support"
<PriorityLevel> ::= "High" | "Medium" | "Low"
<Outcome> ::= <sentence>

;---------------------------------------------------------------
; CONFIDENCE & AFFECT — cognitive and emotional transparency
; Ammann 2025 (affective dimension) + Zimmermann 2024 (challenges: cognitive overload).
; Tentina R2–R4 emphasize explainability and validation to build trust.
;---------------------------------------------------------------
<ConfidenceAndAffect> ::= 
    "Confidence Level: " <confidenceScore> "% (" <confidenceType> ")\n"
    "User Guidance: " <AffectCue> "\n"
    "Interpretation Note: " <AffectExplanation> "\n"

<confidenceType> ::= "Derived from data variance" | "Heuristic" | "Model-based"
<AffectCue> ::= "High trust" | "Caution advised" | "Needs validation"
<AffectExplanation> ::= <sentence>

;---------------------------------------------------------------
; REFERENCES — provenance and traceability
; Tentina F2 (pipeline visibility) + Mendling 2021 (CogniDia: semantic transparency).
;---------------------------------------------------------------
<References> ::= 
    "Source Log: " <LogID> " (" <systemOrigin> ")" "\n"
    "Version: " <version> "\n"
    "Last Update: " <datetime> "\n"
    "Analyst/Producer Contact: " <contact>
```

## Examples

### Short 
```yaml
Insight ID: O2C-017
Process Area: Order-to-Cash
Business Goal: Reduce invoice waiting time between order approval and invoice generation
Analysis Type: Directed
Relevance: 35% delay increase in waiting time from baseline

Observed Behavior: Average delay between approval and invoicing increased from 48h to 65h.
Underlying Data Scope: Event log: O2C_2025_Q3, timeframe: 2025-07-01–2025-09-30
Applied Filters / Pipeline Notes: Only approved orders with invoice issued; excluded canceled orders.
Context: This step occurs after Finance validation and before invoice posting.

Summary: Waiting time threshold exceeded for 312 cases (12% of total volume).
Impact: +17h average delay (affects throughput and cash flow)
Key Takeaway: Delay originates mainly from manual invoice approval steps in Finance.

Recommended Actions:
- Automate invoice approval workflow (Responsible: Finance Manager, Priority: High). Expected Outcome: Reduce average waiting time by 30%.
- Reassign low-risk invoices to automatic processing (Responsible: IT Automation Lead, Priority: Medium).

Confidence Level: 92% (Derived from data variance)
User Guidance: High trust
Interpretation Note: Based on stable sample of 2,600 cases; no missing timestamps detected.

Source Log: O2C_2025_Q3 (ERP)
Version: v1.4
Last Update: 2025-10-01
Analyst/Producer Contact: a.brunner@company.com
```

### Extended
```yaml
Insight ID: ITSM-EXP-04
Process Area: IT Service Management – Incident Resolution
Business Goal: Identify inefficiencies in ticket resolution and understand root causes of rework loops
Analysis Type: Exploratory
Relevance: 22% of incidents contain at least one rework loop

Observed Behavior: High-frequency back-and-forth transitions between “In Progress” → “Pending Info” → “In Progress”.
Underlying Data Scope: Event log: ITSM_2025_GLOBAL, timeframe: 2025-01-01–2025-09-30
Applied Filters / Pipeline Notes: Excluded automatically closed tickets; merged duplicate event labels; retained only Tier 1 incidents.
Context: These loops mainly occur in customer support interactions when incomplete information is provided.

Summary: 2,430 of 10,980 cases show recurring rework patterns causing +3.8 days mean delay.
Impact: Affects service level compliance (SLA breach risk: +18%).
Key Takeaway: Missing initial diagnostic data from customers triggers repetitive clarification cycles.

Analyst Notes: Process mining variant explorer reveals strong clustering around variant 8. (“reassign–pending–update–pending–close”), with 61% of rework loops involving reassignment between Tier 1 and Tier 2 teams.
Linked KPIs: Mean Resolution Time, SLA Violation %, Ticket Touchpoints Count
Supporting Examples: Reference dashboards/examples: Global_ITSM_Rework_2024, Tier2_Pilot_2025.

Recommended Actions:
- Introduce mandatory “Initial Diagnostic Form” for new incidents (Responsible: Service Desk Manager, Priority: High). Expected Outcome: Reduce rework loops by 40%.
- Implement automatic rule: prevent reassignment unless all diagnostic fields are filled (Responsible: System Owner, Priority: Medium).
- Review top 5 recurring categories (“Network”, “Access Rights”, “Hardware”) for additional training content (Responsible: Knowledge Management Lead, Priority: Medium).

Confidence Level: 85% (Model-based)
User Guidance: Caution advised
Interpretation Note: Exploratory result; some cases lack complete timestamps due to regional data sync lag.

Source Log: ITSM_2025_GLOBAL (ServiceNow)
Version: v2.7
Last Update: 2025-10-15
Analyst/Producer Contact: irene.liu@company.com

```