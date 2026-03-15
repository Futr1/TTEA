# Architecture

This platform uses a layered TTEA-inspired system layout:

1. `GlobalObjective`
   - Computes system utility from stability, efficiency, resource pressure, and local task rewards.
2. `SystemImpactAssessment`
   - Estimates short-term utility deltas for candidate actions.
3. `Hierarchy`
   - One global leader
   - Multiple category leaders
   - Multiple individual agents per category
4. `Evolution`
   - Skill reinforcement
   - Learning decision support
   - Long-term utility tracking and elimination
5. `Association`
   - Observation encoder
   - Macro and micro adapters
   - Vector-text bridge
   - Cross-class memory exchange
6. `TaskGroup runners`
   - Web navigation
   - Translation
   - Knowledge enhancement

The platform is intentionally config-driven. Paper-specific assumptions that were not explicitly defined are encoded in [configs/platform.json](d:\文件\论文\第二篇论文-etta多智能体\code\configs\platform.json) and per-experiment JSON files under [configs/experiments](d:\文件\论文\第二篇论文-etta多智能体\code\configs\experiments).
