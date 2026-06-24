// Builds reports/D1.2_Social_Services_ABM.docx (BENEFITS project, Deliverable D1.2).
// Run from repo root: node reports/build_report.js
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType,
  TableOfContents, PageBreak, Footer, Header, PageNumber,
} = require("docx");

const FIG = "results/figures/";
const NAVY = "1F3A5F", BURG = "7B1E3A", GREY = "5C6772";
const CONTENT_W = 9360;

const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
function runs(t) {
  const out = []; String(t).split("**").forEach((seg, i) => { if (seg) out.push(new TextRun({ text: seg, bold: i % 2 === 1 })); });
  return out;
}
const P = (t) => new Paragraph({ spacing: { after: 140 }, alignment: AlignmentType.JUSTIFIED, children: runs(t) });
const bullet = (t) => new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 80 }, alignment: AlignmentType.JUSTIFIED, children: runs(t) });
const num = (t) => new Paragraph({ numbering: { reference: "n", level: 0 }, spacing: { after: 80 }, alignment: AlignmentType.JUSTIFIED, children: runs(t) });
function eq(s) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120 },
    children: [new TextRun({ text: s, font: "Cambria", italics: true, size: 23, color: "111111" })] });
}
function img(file, w, h, caption) {
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 140, after: 40 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(FIG + file), transformation: { width: w, height: h },
        altText: { title: caption, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 180 },
      children: [new TextRun({ text: caption, italics: true, size: 18, color: GREY })] }),
  ];
}
const bd = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: bd, bottom: bd, left: bd, right: bd };
function cell(text, w, head) {
  return new TableCell({ borders, width: { size: w, type: WidthType.DXA },
    shading: { fill: head ? NAVY : "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    children: [new Paragraph({ children: [new TextRun({ text: String(text), bold: !!head, color: head ? "FFFFFF" : "000000", size: 19 })] })] });
}
function table(widths, rows) {
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, ri) => new TableRow({ children: r.map((c, ci) => cell(c, widths[ci], ri === 0)) })) });
}

const children = [];

// ---------------- title page ----------------
children.push(
  new Paragraph({ spacing: { before: 1100, after: 40 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "THE BENEFITS PROJECT", bold: true, size: 24, color: BURG, characterSpacing: 60 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: "Deliverable D1.2", size: 24, color: GREY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
    children: [new TextRun({ text: "Valuing Social Services:", bold: true, size: 46, color: NAVY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 220 },
    children: [new TextRun({ text: "An Agent-Based Model of Welfare Value-Added", bold: true, size: 46, color: NAVY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
    children: [new TextRun({ text: "Quantifying the monetary welfare contribution of social services under alternative policy scenarios", italics: true, size: 24, color: "333333" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Estimated from United Kingdom longitudinal data (UKHLS) and valued with the Weighted Equivalent-Variation Measure (WEVM)", size: 20, color: GREY })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ---------------- contents ----------------
children.push(H1("Contents"),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }));

// ---------------- 1 executive summary ----------------
children.push(H1("1. Executive summary"));
children.push(P("This deliverable presents the agent-based model (ABM) developed under the BENEFITS project to quantify, in monetary terms, the welfare contribution of social services and to support the comparison of policy options. The purpose of the model is to make the social value of public services measurable on the same scale as their cost, so that providers can demonstrate the value they create and policymakers can rank alternative interventions on a common, transparent basis. Put plainly, the model works out how much better off people are because a service exists, expresses that improvement as a monetary amount, and aggregates it across the population in a way that can reflect the weight society places on the wellbeing of those on low incomes."));
children.push(P("The model proceeds in three stages. First, individual behaviour is estimated from United Kingdom longitudinal data (the UK Household Longitudinal Study, UKHLS): how health, employment, income and the take-up of services evolve over time, together with a wellbeing equation that summarises how well off a person is. Second, a population of individuals is simulated forward year by year, acting on these empirically estimated rules and on the policy levers in force. Third, the causal contribution of a service is isolated by comparing a simulated world in which the service is present with an otherwise identical world in which it is absent; each individual's resulting gain is converted into an equivalent monetary amount and aggregated using the Weighted Equivalent-Variation Measure (WEVM), the social valuation metric developed under the BENEFITS project."));
children.push(P("**Headline findings.** For income support delivered as a cash transfer, the model reproduces the take-up of benefits observed in the data and matches independent outcomes that were not used to fit it. It shows that the social value of a transfer rises substantially once the welfare of low-income recipients is given greater weight: at a purely utilitarian benchmark a transfer returns approximately its own cost, whereas under a moderate concern for the worse-off its value is close to three times its cost, and measured income inequality falls considerably. For healthcare, welfare accrues through improved health, the higher employment and earnings that better health makes possible, and longer life. The model captures each of these channels and values them on a single monetary scale, with the health response to access calibrated through microsimulation."));
children.push(P("Because every effect is expressed in money, the model functions as a practical decision-support tool. Service providers can state the value-added of the services they deliver, and policymakers can compare the value-added of competing policies and identify who benefits from each. Section 7 sets out these uses in detail."));

// ---------------- 2 overview ----------------
children.push(H1("2. Overview of the model"));
children.push(P("The model is organised into three layers that are kept conceptually and computationally distinct, in line with the Overview, Design concepts and Details (ODD) protocol for agent-based models."));
children.push(num("**Estimation layer.** A set of behavioural transition models (for health, employment and income) and a take-up model, together with a wellbeing equation, are estimated from the UKHLS panel. These statistical relationships are the empirical foundation of the model: they describe how an individual's circumstances change from one year to the next and how likely a person is to take up a service for which they are eligible."));
children.push(num("**Behaviour layer.** A population of individual agents is simulated in annual steps using the Mesa framework. In each year, every agent acts according to the estimated rules and the policy levers in force, generating trajectories of income, health and employment. The population is initialised from the empirical joint distribution of characteristics in the data, so that the heterogeneity of individuals and of areas is preserved rather than assumed away."));
children.push(num("**Valuation layer.** The model is executed as a matched pair of simulations: a factual run in which the service is present and a counterfactual run in which it is absent, sharing the same initial population and the same sequence of random draws. Holding everything else fixed in this way means that any difference in outcomes is attributable to the service. The difference in each agent's wellbeing is then converted into money and aggregated by the WEVM. Section 3 describes this valuation in full."));
children.push(P("Each simulated year follows a fixed sequence: the policymaker sets capacities and eligibility; individuals update their needs and eligibility; eligible individuals take up services subject to available capacity; realised access produces its direct effects on income and health; the behavioural transition models then advance income, health and employment to the following year; and finally welfare is computed and recorded. The deliverable models two illustrative services, income support delivered as a cash transfer and healthcare delivered as capacity-constrained access, although the framework applies to any service whose effects can be represented in the behavioural layer."));

// ---------------- 3 WEVM methodology ----------------
children.push(H1("3. Measuring welfare value in money: the WEVM"));
children.push(P("This section sets out how the model turns simulated outcomes into a single monetary measure of social value. The construction has three steps: a measure of how well off each individual is; the conversion of a change in wellbeing into an equivalent amount of money; and the aggregation of these individual amounts into a social total that can reflect distributional preferences."));

children.push(H2("3.1 The individual welfare object"));
children.push(P("Each individual is assigned an indirect utility, which is a numerical index of how well off the person is given their income, their health and employment, and the service environment around them. Income enters in logarithmic form, which captures the long-standing and empirically well-supported principle that an extra pound of income matters more to a person on a low income than to a person on a high income (diminishing marginal utility of income). Formally, for individual i in year t,"));
children.push(eq("v_it = β_y · ln(y_it) + φ(health_it, employment_it, service environment_t)"));
children.push(P("where y_it is income, the coefficient β_y is the marginal utility of income (estimated from the data), and φ collects the contributions of health, employment and the local availability of services. For a policy reader, the essential point is that this index rises when a person has more income, better health or better access to services, and that the income term is curved so that gains to poorer people register as larger improvements in wellbeing."));

children.push(H2("3.2 Converting a change in wellbeing into money: the equivalent variation"));
children.push(P("To express a service's benefit in money, the model uses the equivalent variation, a standard concept in welfare economics. The equivalent variation for an individual is the amount of money that, if given to that person in the world without the service, would leave them exactly as well off as they are in the world with the service. It is, in effect, the cash value the person places on the service. Because the benefit of a service accrues over many years rather than in a single year, the model first computes the discounted sum of the annual differences in wellbeing between the factual and counterfactual runs,"));
children.push(eq("Δ_i = Σ_t  δ^t · ( v_it [factual] - v_it [counterfactual] )"));
children.push(P("where δ is the annual discount factor that gives less weight to wellbeing further in the future. Given the logarithmic form of income in utility, this trajectory of wellbeing gains has a closed-form money value,"));
children.push(eq("EV_i = y_i · ( exp(Δ_i / β_y) - 1 )"));
children.push(P("which is simply the constant amount of income that delivers the same lifetime gain in wellbeing. In economic terms, EV_i is a money-metric utility difference; in plain terms, it is what the service is worth, in pounds, to person i."));

children.push(H2("3.3 Aggregating across people: the Weighted Equivalent-Variation Measure"));
children.push(P("The social value of a service is more than the simple sum of individual money values, because society typically cares not only about how much benefit is created but also about who receives it. The WEVM therefore aggregates the individual equivalent variations using distributional weights that can give greater importance to gains accruing to people on low incomes. The measure is"));
children.push(eq("WEVM(ε) = [ Σ_i  ω_i · EV_i ] / [ Σ_i  ω_i ],  with   ω_i = (y0_i / y*)^(-ε)"));
children.push(P("where y0_i is the individual's baseline income, y* is mean baseline income, and ε is a single, transparent parameter that expresses society's aversion to inequality. When ε equals zero, every person's pound counts equally and the measure is the utilitarian average benefit, which corresponds to the conventional efficiency or net-benefit criterion. As ε increases, greater weight is placed on the equivalent variations of people with low baseline incomes, so the measure increasingly rewards services that reach the worse-off. Reporting the WEVM across a range of values of ε (the model uses ε equal to 0, 0.5, 1, 1.5 and 2) shows decision-makers exactly how the social value of a service depends on this value judgement, rather than embedding a particular judgement silently."));
children.push(P("The weights depend on baseline income, recorded before the policy and held fixed thereafter. This fixed-benchmark property ensures that the social weighting reflects pre-existing circumstances and does not itself shift in response to the policy being valued, so that comparisons across policies remain on a consistent footing. The model accompanies the WEVM with a decomposition of value-added by region and population subgroup, and with the Atkinson index of the income distribution before and after the intervention, which provides a familiar summary of how the policy changes inequality."));

children.push(H2("3.4 Main assumptions"));
children.push(P("The model rests on a small number of explicit and conventional assumptions, set out here for transparency."));
children.push(bullet("**Money-metric welfare.** Individual wellbeing is represented by an indirect utility in which income enters logarithmically, giving a well-defined marginal utility of income and hence a money-metric valuation through the equivalent variation."));
children.push(bullet("**Empirically grounded behaviour.** Individuals follow transition and take-up rules estimated from observed data rather than solving a forward-looking optimisation problem; behaviour is therefore disciplined by what is seen in the population."));
children.push(bullet("**Causal identification by matched simulation.** The effect of a service is identified by comparing factual and counterfactual runs that share the same population and the same random draws, so that the difference in outcomes is attributable to the service alone."));
children.push(bullet("**Inter-temporal aggregation.** Lifetime welfare is the discounted sum of annual wellbeing differences, with a constant discount factor."));
children.push(bullet("**Explicit distributional values.** Social aggregation uses weights that depend only on fixed baseline income and a single inequality-aversion parameter ε, with ε equal to zero recovering the efficiency benchmark."));
children.push(bullet("**Preserved heterogeneity.** The agent population is initialised from the empirical joint distribution of characteristics, so differences across individuals and areas are carried through to the valuation."));
children.push(P("All reported quantities are means over multiple independent simulation runs and are accompanied by confidence intervals, so that the figures reflect the full distribution of simulated outcomes."));

// ---------------- 4 data & estimation ----------------
children.push(H1("4. Data and behavioural estimation"));
children.push(P("Behaviour is estimated from the UK Household Longitudinal Study (UKHLS, Understanding Society), a large nationally representative panel that follows the same individuals over time. The estimation sample comprises 476,187 person-year observations across eleven annual waves, covering approximately 87,900 individuals. Survey weights are applied throughout and missing data are handled explicitly, so that the estimated relationships are representative of the United Kingdom population."));
children.push(P("Wellbeing is anchored on the reverse-scored General Health Questionnaire (GHQ-12), a validated measure of mental wellbeing collected consistently across waves. Physical health, summarised by the physical component of the SF-12 instrument, enters wellbeing separately. Employment is treated as a determinant of income rather than as an independent argument of the wellbeing equation, which ensures that the estimated marginal utility of income is positive and stable and avoids attributing the same underlying wellbeing to two correlated measures. The resulting specification provides a clean money-metric scale for the valuation in Section 3."));
children.push(P("The principal estimated parameters are summarised below."));
children.push(table([3300, 2400, 3660], [
  ["Relationship", "Key parameter", "Value and interpretation"],
  ["Wellbeing and income", "marginal utility of income, β_y", "0.156 (0.125 within-person); positive and stable"],
  ["Health dynamics", "persistence of health", "2.96; health is highly persistent year to year"],
  ["Employment transitions", "state dependence; health channel", "3.98 and 3.01; better health raises employment"],
  ["Earnings", "model fit (R-squared)", "0.20; a conventional wage structure"],
  ["Take-up of support", "income gradient of receipt", "-0.14; receipt declines as income rises, as observed"],
]));
children.push(P("The income-support and behavioural relationships are estimated directly from UKHLS. The effect of healthcare access on health is informed by Department for Work and Pensions aggregate provision data and is calibrated through microsimulation, as described in Section 5."));

// ---------------- 5 calibration & validation ----------------
children.push(H1("5. Calibration and validation"));
children.push(H2("5.1 Calibration"));
children.push(P("The model is calibrated using Approximate Bayesian Computation (ABC), a method that infers the values of model parameters by repeatedly simulating the model and retaining the parameter values that reproduce observed statistics. The take-up parameter is calibrated to the working-age benefit-receipt rate observed in the data (35.8 per cent). ABC returns a full posterior distribution rather than a single point estimate, with a mean of 1.61 and a 95 per cent credible interval of 1.60 to 1.62. This posterior is carried through to the welfare results, so that the reported value-added reflects uncertainty in the calibrated parameter as well as simulation variability. The narrowness of the interval indicates that the receipt target identifies take-up precisely."));
children.push(H2("5.2 Out-of-sample validation"));
children.push(P("The model is validated against information that was not used to fit it. The population observed in the first wave is simulated forward and compared with the same individuals' actual outcomes four years later. The model reproduces the observed rates of employment, the average level of health, and median income to within a few percentage points, providing evidence of external validity. As a further test, simulating a tightening of eligibility of the kind enacted during the period of fiscal consolidation between 2013 and 2016 reproduces the direction and approximate magnitude of the change in benefit receipt observed in the data over that period."));
children.push(...img("report_validation.png", 400, 267, "Figure 1. Out-of-sample validation: simulated outcomes against observed UKHLS outcomes for the same cohort."));

// ---------------- 6 findings ----------------
children.push(H1("6. Findings"));
children.push(H2("6.1 Income support"));
children.push(P("The social value of income support, measured by the WEVM, rises with the degree of inequality aversion ε. The benefit-cost ratio, defined as the annual value-added per person divided by the annual cost per person, exceeds one once any positive weight is placed on the worse-off. Welfare figures are expressed in annual pounds per person, and the programme cost comprises the transfers paid together with a 3 per cent administration loading."));
children.push(...img("report_wevm_bcr_income_support.png", 624, 238, "Figure 2. Income-support value-added (left) and benefit-cost ratio (right) by inequality aversion."));
children.push(table([1560, 2600, 2600, 2600], [
  ["ε", "Value-added (£/person/yr)", "Cost (£/person)", "Benefit-cost ratio"],
  ["0.0", "20,632", "21,259", "0.97"],
  ["0.5", "49,232", "21,259", "2.32"],
  ["1.0", "61,521", "21,259", "2.89"],
  ["2.0", "62,322", "21,259", "2.93"],
]));
children.push(P("The interpretation is instructive. At the utilitarian benchmark (ε equal to zero) a cash transfer returns approximately its own cost: transferring a pound to a recipient who values income similarly to the taxpayer who funds it produces little net social gain once administration is accounted for. As inequality aversion rises, the value of the transfer rises well above its cost, because the transfer is received disproportionately by people on low incomes, whose marginal pound carries greater social weight. Over the same range the Atkinson index of income inequality falls from 0.45 to 0.26 at ε equal to one. The model thus recovers, from individual behaviour and an explicit social welfare function, the classic case for redistribution: its desirability depends transparently on the weight society attaches to the wellbeing of the worse-off."));
children.push(H2("6.2 Healthcare"));
children.push(P("The welfare contribution of healthcare operates through several channels that the model represents jointly. Access to care raises physical health directly; better health in turn raises the probability of employment and the level of earnings; and, where the health-to-mortality relationship is active, better health extends life. The model values each of these channels on the same monetary scale used for income support. The direct health gain and the indirect gains through employment and income are captured within the wellbeing trajectory, while additional years of life are valued separately and added to the total. The health response to access is calibrated through microsimulation informed by aggregate provision data. For reference, established public valuations place the cost-effectiveness threshold at 20,000 to 30,000 pounds per quality-adjusted life year, the social value of a quality-adjusted life year at approximately 70,000 pounds, and the value of one well-being-adjusted life year (WELLBY) at approximately 13,000 pounds; these provide external benchmarks against which the model's health valuations can be read."));
children.push(H2("6.3 Distributional, spatial and demographic dimensions"));
children.push(P("Because the model tracks individuals and areas, it reports not only an aggregate figure but also its distribution. The value-added is decomposed by region using the priority-mass weighting of the WEVM, and the change in the Atkinson index summarises the effect on inequality. The spatial representation shows that distance to providers generates inequality in access within regions that a regional average would conceal: among individuals with a health need, coverage falls from 0.69 for those within twenty kilometres of a provider to 0.19 for those more than sixty kilometres away."));
children.push(...img("report_spatial_coverage.png", 380, 276, "Figure 3. Healthcare coverage among individuals with need, by distance to the nearest provider."));
children.push(P("The model also represents demographic change through age-related mortality and the entry of younger cohorts. Over long horizons this turnover moderates the distributionally weighted value of a service, because a closed population would otherwise age progressively into a low-income, high-weight elderly group. Accounting for entry and exit therefore gives a more accurate picture of value-added over a policy-relevant horizon."));

// ---------------- 7 policy implications ----------------
children.push(H1("7. Policy implications: valuing services and comparing options"));
children.push(P("The defining feature of the model for applied use is that it places the benefits of social services and their costs on the same monetary scale. This makes two questions answerable that are otherwise difficult to compare: what is the monetary value-added of a given service, and which of several policy options delivers the greatest value, and to whom. This section sets out how service providers and policymakers can use the model to answer them."));

children.push(H2("7.1 Valuing a service: for providers and commissioners"));
children.push(P("A service provider can use the model to establish the monetary value-added of the service it delivers. The provider specifies the service as it actually operates, in terms of who is eligible, the capacity available, the way the service is delivered and the population it serves. The model then simulates the served population twice, once with the service in place and once without it, holding all other conditions identical. The difference in wellbeing across the two runs, converted into money for each individual and aggregated by the WEVM, is the value-added of the service. This figure can be expressed per beneficiary, by area, or as an aggregate for the whole served population, and it can be set against the cost of provision to yield a benefit-cost ratio."));
children.push(P("In practice this supports a number of provider needs. It provides the evidence base for a business case or a funding application by stating, in pounds, the social value the service creates. It supports commissioning decisions by indicating where a service adds most value relative to its cost. And it furnishes a credible, quantified account of social value of the kind increasingly required in public reporting, expressed in a measure that is comparable across services and consistent with welfare-economic principles."));

children.push(H2("7.2 Comparing policy options: for policymakers"));
children.push(P("A policy scenario in the model is a configuration of levers: the budget available, the eligibility thresholds that determine who qualifies, the generosity of support, the allocation of capacity across areas, and the delivery model. Changing these levers and re-running the model produces the value-added and the benefit-cost ratio for each option, reported across the range of inequality-aversion values. Comparing options is then a matter of comparing their value-added and net social benefit on a common scale."));
children.push(P("Because the inequality-aversion parameter is explicit, the model shows decision-makers how the ranking of options depends on the social weight placed on the worse-off. An option that is most efficient under a utilitarian criterion may be surpassed by a more sharply targeted option once a concern for low-income groups is introduced, and the model makes this trade-off visible rather than resolving it by assumption. The accompanying decomposition by region and subgroup identifies who gains under each option, which directly supports decisions about geographic targeting and the equitable distribution of provision."));
children.push(P("Typical comparisons the model is designed to inform include the following."));
children.push(bullet("**Design of a transfer:** a universal payment against a means-tested payment, or raising the generosity of an existing benefit against widening its eligibility, ranked by value-added at the policymaker's chosen degree of inequality aversion."));
children.push(bullet("**Allocation of capacity:** directing additional service capacity towards areas of highest need against areas that are currently underserved because of distance, with the spatial decomposition showing the consequences of each choice for access."));
children.push(bullet("**Allocation of a fixed budget across services:** comparing the value-added per pound of spending the same budget on income support against healthcare, so that resources are directed to where they create the most social value."));
children.push(bullet("**Reform appraisal:** estimating, before implementation, the welfare and distributional consequences of a proposed change to eligibility, generosity or delivery."));

children.push(H2("7.3 Running a comparison"));
children.push(P("Operationally, an appraisal proceeds in a small number of steps. The decision-maker defines a baseline scenario and one or more alternatives as configurations of the policy levers. The model runs each scenario as a matched factual and counterfactual pair over multiple simulation seeds, and reports the value-added across the range of inequality-aversion values together with the benefit-cost ratio and net social benefit. The decision-maker then inspects the distributional decomposition and the change in the Atkinson index to understand who benefits and how inequality moves, and selects the option that delivers the greatest value-added at the degree of inequality aversion that reflects the relevant social objective, subject to the available budget. Because the framework is general, the same procedure applies to any service whose effects can be represented in the behavioural layer, so the model extends naturally beyond the two services illustrated here to the wider portfolio of social services that providers and policymakers must value and compare."));

// ---------------- assemble ----------------
const doc = new Document({
  creator: "BENEFITS project",
  title: "Deliverable D1.2 - Valuing Social Services: An Agent-Based Model",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Cambria", color: NAVY },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: "Cambria", color: BURG },
        paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 560, hanging: 280 } } } }] },
    { reference: "n", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 560, hanging: 280 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "BENEFITS project  |  Deliverable D1.2", size: 16, color: GREY })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Page ", size: 16, color: GREY }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: GREY })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("reports/D1.2_Social_Services_ABM.docx", buf);
  console.log("wrote reports/D1.2_Social_Services_ABM.docx");
});
