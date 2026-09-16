# 3-minute video script

**Autonomous Decentralised Search and Coverage with Unmanned Aerial Systems Swarms**
Mahadev Reddy Devireddy Venkata — 2823511

Eight slides. Timings below assume a normal pace; time yourself once before
recording. If you run long, drop the line marked `[CUT]` on slide 7.

Every line is also in the PowerPoint speaker notes, so you can read from
Presenter View.

**The animation on slide 5 only plays in Slideshow mode.** It won't move in the
editing view, and it won't move in a PDF export. Start the slideshow before you
start recording.

---

## Slide 1 — Title · 0:00–0:11

> Six drones searching an area nobody has mapped. Nothing telling them where to
> go, and no radio between them. I wanted to know whether a swarm that learns can
> cover as much ground as the methods people already use.

*Pause on the title. Don't read your own name — it's on the slide.*

---

## Slide 2 — The task · 0:11–0:34

> A hundred and fifty metre block, six drones, seven hundred steps. This is a real
> run, about a hundred steps in.
>
> Each drone only sees that small box — twenty-one metres out of a hundred and
> fifty. And they can't talk to each other.

*Point at the small box when you mention it.*

---

## Slide 3 — What I built · 0:34–0:57

> I tested four controllers, from none of them learning to all six learning. Two
> classical ones, then a hybrid where only one drone learns, then mine, where all
> six share one network.
>
> Sixteen hundred runs. Every controller got the exact same layout on every seed,
> so I could compare them run against run.

*Slow down on "the exact same layout" — that's the methodological point.*

---

## Slide 4 — The result · 0:57–1:26

> This is the main result, and the two lines cross.
>
> Mine clears the ninety percent target at every obstacle density, on both sets of
> layouts. So does potential fields.
>
> In open ground mine wins by about five points. Once obstacles appear I can't
> separate them statistically, and at the hardest density potential fields is
> slightly ahead.
>
> So it matches the classical controller. It doesn't beat it everywhere.

*The last line matters. Say it plainly.*

---

## Slide 5 — What that looks like · 1:26–1:46

> Here they are on the same layout, from the same starting point.
>
> Watch the classical swarm funnel along the gaps between buildings. Mine pushes
> outward and splits the area up — and none of those six is telling the others
> where it's going.

*Let it run. Stop talking for the last few seconds and let them watch.*

---

## Slide 6 — The finding that mattered · 1:46–2:16

> The thing I found most useful wasn't in that comparison at all.
>
> I kept the network, the budget, the seed and everything the drones see the same,
> and changed one thing: what they're rewarded for. Reward each drone for the
> ground it covers first, and it works. Reward them on the team total, and coverage
> drops thirty point six points — not one run in fifty reaches the target.
>
> That's a bigger drop than the gap between any two controllers I tested.

*Let "thirty point six" land. One beat of silence after it.*

---

## Slide 7 — What I found against myself · 2:16–2:42

> I also checked my own experiments, and some of it went against me.
>
> An earlier draft reported a number I couldn't reproduce. The model that survived
> scores ninety, not ninety-five, which reverses what I'd concluded.
>
> [CUT] Four training seeds showed I'd stopped training too early.
>
> And potential fields only looks like it improves as obstacles increase — counting
> actual cells, it covers six hundred and forty fewer.

*Steady, not apologetic. This slide is why a marker trusts the rest.*

---

## Slide 8 — Close · 2:42–2:56

> So: a swarm that learns, with no messages between the drones, matches a
> well-tuned classical controller and beats it in open ground. What I rewarded them
> for mattered more than the network. And measuring coverage as a percentage
> misleads whenever the open ground changes.
>
> Thank you.

---

## Delivery notes

**Don't read the slides.** Every slide has its numbers on it. Say the sentence
*around* the number and let the viewer read the number.

**Slide 5 is the one to rehearse.** The animation runs about seven and a half
seconds per loop. Talk over the first loop, then stop and let it play. Silence
while something is moving reads as confidence; talking over it reads as nerves.

**Recording.** PowerPoint's Record Slide Show, or Zoom screen-share to a local
file. Target 1080p — three minutes usually lands at 25–40 MB, inside the limit. If
it comes out bigger, export at 720p.

**One thing to avoid.** Don't oversell. The report's strength is that its claims
match its evidence, and the video is marked *in combination* with it. If the video
claims more than the report does, that mismatch is what a marker will notice.
