A Trust Test for Data-Centre Energy Controllers

What the project discovered, why it matters, and what should happen next

**Project:** Data-Centre Energy-Control Claim Qualification Benchmark

**Author:** Babasola Osibo

The project in one sentence

This project created a demanding test for software that coordinates data-centre electricity, renewable generation, batteries, local generation, and flexible computing work—so that promising cost results are not mistaken for dependable performance.

The headline finding

**Average cost alone gave an incomplete picture.** Many reinforcement-learning controllers produced promising averages, but far fewer delivered the required service consistently across repeated training runs.

The main study trained four reinforcement-learning methods across 12 energy-system designs. Each method and design was trained five separate times, producing 240 controllers. Every controller was then tested on the same 103 weeks from 2024–2025 that had not been used for training.

- 136 of 240 controllers completed the test with no recorded service failure.
- Only **4 of 48 method-and-system combinations** passed in all five repeated training runs.
- The other 44 combinations could not support the planned headline cost comparison because at least one repeat failed the service requirement.

This does **not** show that reinforcement learning cannot help. It shows that an attractive average from selected runs is not enough to establish a repeatable, service-qualified advantage.

What the benchmark adds

The benchmark checks issues that can otherwise make a controller appear better than it really is. It requires every controller to face the same weeks and service rules. It also accounts for unfinished computing work and remaining battery energy at the end of each week, preventing a controller from looking cheaper simply by postponing obligations beyond the test.

It also compares controllers with a same-model hindsight reference that knows the full future week. That reference is not deployable; it is used only to show how much modeled opportunity remains.

The practical contribution is a **trust test for performance claims**. It can expose fragile results before an organization commits further time and money to deployment, while giving successful future controllers a clearer and more credible standard to meet.

A constructive signal

A separate experiment tested whether better—but still causal—price information could help a simple fixed controller. Using a summary of the previous week’s prices improved modeled cost in every tested system design while preserving zero recorded service failures in both versions.

Across the study’s combined calculation, this closed **about 7% of the measured gap to the hindsight reference**. It is a useful research signal, not a claim of 7% operational savings and not evidence of production performance.

Another 100-controller study tested recurrence and longer training. Half of those controllers passed individually, but none of the six planned comparison groups was complete. More model complexity and more training therefore did not reliably solve the service-consistency problem in these experiments.

What to take from this

Before investing heavily in an energy controller, ask whether its claimed benefit survives repeated training, strict service requirements, causal information, and complete end-of-period accounting. A controller that cannot pass those checks consistently is not yet supported by strong deployment evidence, even when its average cost looks attractive.

The next research step is not simply to train longer. It is to design controllers around service qualification, test useful information inputs, and then repeat the evaluation prospectively and in additional regions and operating conditions.

Boundaries

These are simulation results for the tested algorithms, five training repeats, predefined energy-system designs, and one ERCOT-centred facility model. They do not establish production reliability or savings for a particular operator. Independent audits reproduced the main corrected and history/budget conclusions from preserved evidence without rerunning every simulation. The price-information result has a complete numerical replay and is reported only as a descriptive fixed-policy ablation, while a reserved future-period confirmation remains outstanding.
