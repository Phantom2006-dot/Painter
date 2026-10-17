## Recommendation

Replace the YOLO box detector as the source of text geometry, and invert the pipeline so OCR runs first. Use a segmentation-based text detector that emits **polygons** (DBNet++ or FCENet), straighten each detected text line with a thin-plate-spline warp before recognising it, and only then recover layout, padding, margin and border by geometric reasoning over the text fragments. Keep DocLayout-YOLO, but demote it to finding large non-text structures such as tables, figures and rule-delimited regions, where an axis-aligned box is genuinely the right answer. Add UVDoc as a dense page-unwarping stage so the page itself is flattened before any of this runs.

The reason is short: a rectangle is the wrong shape, and no amount of training changes that. Your instinct to put OCR first is correct, and the humanities OCR world solved the bending problem years ago — it just calls it a *baseline* rather than a text box.

## Why axis-aligned boxes cannot be fixed by training

A YOLO detector's regression head predicts a centre, a width and a height. That output format has four degrees of freedom and no way to express curvature. When a page curls, a bent line of text has corner points that are not on any rectangle, so the detector has to choose: cover the whole bent line and swallow the background above and below it, or cover the dense middle and clip both ends. Both are wrong, and which one it picks varies with the curl. This is why the failures look erratic rather than consistently bad.

The DBNet paper states the same thing from the other side: segmentation-based methods became popular specifically because "the segmentation results can more accurately describe scene text of various shapes such as curve text" [3]. That sentence is the whole argument. A pixel mask has no shape constraint, so it can follow a curve. A rectangle does.

## What genuinely bends

These are the representations that can follow bent text, roughly from most to least constrained.

**TextSnake** represents a text instance as "a sequence of ordered, overlapping disks centered at symmetric axes", where each disk carries a radius and an orientation [1]. The predicted centreline follows the direction of the curved text, and the union of the disks reconstructs the bent region. This is the cleanest conceptual model of the problem: the text is a thick curve, not a rectangle.

**ABCNet** and ABCNet v2 fit text with a parameterised cubic Bezier curve — two curves for a text instance — and use a BezierAlign layer to sample convolution features along that curve rather than a rectangle [9]. The authors describe it as fitting "arbitrarily-shaped text by a parameterized Bezier curve" with negligible computational overhead.

**FCENet** goes further and models the text contour in the frequency domain. It predicts a compact Fourier signature, then reconstructs the full contour with an inverse Fourier transform and non-maximum suppression at test time [2]. The paper claims the method is "accurate and robust to fit contours of scene texts even with highly-curved shapes" and reports particular strength on the highly-curved subset of CTW1500 and Total-Text [2]. For a page that is severely curled, this is the strongest of the polygon family.

**DBNet, PSENet and PAN** are pixel-segmentation detectors. They output a probability map that is thresholded and traced into a contour, so the resulting polygon can be curved [3]. DBNet adds a differentiable binarisation step so the threshold is learned with the network rather than tuned by hand [3]. These are the fastest and most practical of the group.

**DRRG** splits a text instance into a series of small rectangular components and then links them with a graph reasoning network, estimating height, width and orientation per component [3]. It handles arbitrary shapes by composition rather than by a curve primitive, which makes it useful when text is fragmented.

The practical point is that all of these output a **polygon** — an ordered list of contour points — and a polygon can describe a bent line. Downstream code that currently consumes `x1, y1, x2, y2` needs to consume a point list instead, which is a contained change.

## The flattening route, which is easier to exploit

There is a second answer that matters more for a document pipeline: make the page flat, and then rectangles become correct again. Document dewarping estimates the deformation of the physical page and inverts it.

**UVDoc** is the most usable option here. It predicts a 3D grid mesh and a 2D unwarping grid with a small fully-convolutional network, is released under the **MIT licence** with the trained weights (`model/best_model.pkl`) committed directly in the repository, and runs from a single command through `demo.py` [6]. There is no licence or download friction. On its own benchmark it reports MS-SSIM 0.785 and AD 0.119, against DocGeoNet at 0.706/0.168, DocTr at 0.697/0.160 and DewarpNet at 0.589/0.193 [6]. Note that those corrected figures are an erratum the authors published after finding the original numbers used only half the benchmark, so quote the corrected set [6].

The rest of that family, in order of how much I would trust them for this use case: **DewarpNet** models the 3D shape of the page explicitly and maps texture onto it; **DocTr** and **DocTr++** use a transformer to predict a dense flow field, which also corrects illumination; **DocGeoNet** and **DocAligner** are later refinements [15]. The **DocUNet** benchmark exists to measure exactly this and is the standard reference point [8]. We already have the DocUNet paired photo-and-scan set downloaded in this project, so evaluation is immediately possible.

Dewarping and polygon detection are complementary rather than alternatives. Dewarping fixes global surface curl, which is the dominant error on a photographed page. Polygon detection catches what remains: text that is still locally curved, rotated in a way the unwarp did not model, or running along a fold that the grid smoothed over.

## The humanities already solved bent-line OCR

This is the finding I would most encourage you to act on, because it is mature, free, and built for exactly this problem.

**Kraken** performs layout analysis by finding **baselines** — "the imaginary line on which text sits" — together with a **bounding polygon** per line, rather than a rectangle [5]. Its output format makes the geometry explicit: an ALTO `TextLine` carries a `BASELINE` attribute (a polyline from the start to the end of the line) and a `<Polygon POINTS="...">` for the line's outline [5]. Both are polylines, so a bent line is represented directly and losslessly.

The pipeline order Kraken documents is precisely the one you described: find the lines and their baselines and determine reading order first, then cut each line out and feed it to the recogniser one at a time, then serialise text and coordinates together [5]. It explicitly notes that historical documents "often have complex layouts with multiple columns, side notes, or drop caps" and that getting the reading order wrong produces "a jumbled word salad" [5]. It also warns that segmentation is where most of the difficulty lives, and that the model must match the material.

The wider ecosystem in this area — **eScriptorium**, **Segmonto**, **dhSegment**, **Transkribus**, **OCRopus** — shares the baseline-and-polygon convention. If you adopt the same representation internally, you can borrow their tooling and their training data instead of rebuilding it.

## Layout recovery from text fragments, without an LLM

Once text fragments and their polygons exist, grouping them is a classical computational geometry problem, and there is a well-documented method for it.

**Kise's area Voronoi diagram method** takes the connected components of the page and builds an area Voronoi diagram over them, using it to establish the spatial relationship between components and to group them bottom-up into words, lines and blocks [4]. The critical property for us: it handles **non-Manhattan layouts**, meaning "irregular or slanted boundaries", and overlapping elements such as a stamp sitting over text [4]. A recent IPOL article provides a faithful, clearly documented **Python-only implementation** that is "readily portable" [4].

The same article is honest about the limits, and you should plan around them. It is "notably sensitive to specific document characteristics, especially in handling complex elements like handwritten text, lists, drop-caps, and tables", the parameters need context-dependent tuning because "there is no single set of default parameters that consistently works well across all cases", and the result depends materially on the binarisation and on border sampling [4]. Treat it as one signal among several rather than a single source of truth.

The classical companions are **Docstrum** (nearest-neighbour clustering of components into lines, then lines into blocks), **XY-cut** (recursive division of the page along projection-profile valleys) [12], and **RLSA** (run-length smoothing, which smears ink runs along rows and columns so that lines and blocks fuse into solid regions that can be extracted as connected components) [11]. A performance comparison of six classic page segmentation algorithms found the three best performers were constrained text-line finding, Docstrum, and the Voronoi-based method [4], which is a useful independent confirmation that this family is the right place to look.

## Computing padding, margin and border from text geometry alone

This is the part you asked for specifically, and it can be done with pure logic. The essential move is to stop thinking of a box as the unit and start thinking of **ink and the gaps between ink**.

Start from the glyph and word polygons produced by the text detector. From these, derive a set of statistics that describe the page's own rhythm: the median inter-word gap, the median inter-line gap measured between baselines, the x-height band of each line, and the median glyph stroke width. Every later decision is expressed as a multiple of these measured quantities, so the method adapts to the document's actual type size instead of assuming one.

**Margin** is then the distance from the outermost text to the page boundary. Determine the page region first — the paper outline you already detect, or the largest bright connected component — then take the convex hull of all text polygons and measure the perpendicular distance from each hull edge to the nearest page edge on that side. The median of those distances per side is the margin for that side, and outliers flag text that sits outside the normal margin, such as a header, a page number or marginalia. An existing approach does exactly this: a published margin computation model "is designed to compute the margin by locating the text area within the document image" and then measuring from the segmented text area outward [13].

**Padding** is the gap between text and the edge of whatever contains it. For a paragraph, take the text polygons belonging to one block, compute the block's own bounding polygon, and the padding is the distance from the tightest text extent to the block's declared boundary. Where the container is a drawn border or a table rule, the padding is simply the distance from the text to that rule — which we can already measure, since the table module detects rule lines. Where the container is only implied, use RLSA with a threshold derived from the line spacing: smearing the ink by roughly half the inter-line gap fuses the paragraph into one region, and the difference between the smeared region and the tight text extent is the padding [11].

**Border** separates into two cases. A drawn border is a rule line, found by the same line-detection used for tables. An implied border is where the whitespace channel around a region is wider than the intra-region gaps — so it falls out of the same gap statistics, as the first gap that exceeds the block's internal gap distribution by a configured factor.

**Line spacing and leading** come from the baseline polylines, not from box heights. This matters because box heights on a curled page are inflated by the curl, while baseline-to-baseline distance measured perpendicular to the baseline is not. Once each line has a baseline, the spacing between consecutive baselines is the leading, and that value feeds the renderer directly.

**Grouping into lines and blocks** is where the Voronoi method and clustering come in. Build the area Voronoi diagram of the word polygons [4], then merge two words into the same line when the gap between them is small relative to the line's own x-height, and when their baselines are close in angle and offset. Merge lines into a block when the gap between them is close to the block's internal leading and their horizontal extents overlap. This handles a two-column page without being told there are two columns, because the gutter is simply a gap that never passes either test. DBSCAN over the word features (baseline angle, baseline offset, gap) is a robust way to implement the same rule, since it needs no preset number of clusters and labels the gutter as noise.

**Reading order** then falls out of the block geometry: order blocks by XY-cut recursion on their polygons, or by sorting on a column index derived from the x-histogram of block centres, then by y within a column. Kraken treats reading order as a first-class output for exactly this reason [5].

None of this needs a language model. It needs measured statistics and explicit thresholds, which is also why it stays debuggable: when padding comes out wrong you can print the gap distribution and see which threshold misfired.

## What is runnable today, on CPU, with open weights

DBNet, PSENet, PANet, FCENet, TextSnake, DRRG and Mask R-CNN are all available as pretrained models in **MMOCR**, with weight download links published in the documentation and Apache-2.0 licensing across OpenMMLab [3]. DBNet with a ResNet-18 backbone reports 62 FPS on the ICDAR2015 benchmark, though that figure is on a GPU; on CPU expect a few hundred milliseconds per page, which is acceptable for an offline pipeline [3].

**UVDoc** is MIT-licensed with weights in the repository and a one-command demo [6].

**Kraken** is an active, freely-licensed engine with released segmentation and recognition models, and a current version 5 paper published in 2025 [5].

**SPTS v2** (ByteDance) treats spotting as sequence prediction from a single-point annotation and reports 19× faster inference than comparable single-point spotters [14]. It is open source, though it is a transformer and therefore heavier on CPU than DBNet.

**Cloud OCR** from Google returns a `BoundingPoly` with an explicit list of vertices and normalised vertices per annotation, rather than a rectangle [7]. That confirms the polygon convention is standard across the industry, and gives you a fallback comparison oracle without adopting it in production.

I would treat ABCNet and TextBPN++ as research-grade: the ideas are excellent and worth reimplementing, but the released code is older and less maintained than MMOCR.

## Staged implementation plan

**Stage A — make the interface accept polygons.** Change the internal detection type from a rectangle to a polygon, keeping a rectangle as the degenerate default so nothing breaks. This is the enabling refactor and it is independent of which model you choose.

**Stage B — swap the text detector.** Run DBNet++ through MMOCR for text, and keep DocLayout-YOLO only for tables, figures and rule-delimited regions. Measure on the curled certificate photograph and on the flat scan, so you can see the curl-specific gain separately from the general change.

**Stage C — rectify lines before recognition.** For each text polygon, skeletonise the mask to get a centreline, then unwarp the line into a straight strip with a thin-plate-spline transform driven by that centreline. Recognise the straightened strip. This is the step that converts "curved text" into "text", and it is what makes the recogniser's job easy again. Compare against recognising the curved line directly to prove the gain.

**Stage D — flatten the page.** Insert UVDoc into preprocessing, after the page-corner quad and before detection. Keep the corner model, because it handles page boundary and orientation, while UVDoc handles interior surface deformation.

**Stage E — replace box layout with geometric grouping.** Introduce the area Voronoi grouping and the gap-statistics method above to derive lines, blocks, padding, margin and leading. At this point the detected boxes stop being the layout, and become merely evidence for it.

**Stage F — only then consider a learned model for the residual.** Once stages A to E are measured, any remaining errors are well characterised, and that is the right moment to consider whether a small learned model or an LLM adds anything.

## Risks and how to hold them

The Kise method's parameter sensitivity is the main technical risk, and it is documented by its own authors: the IPOL article states there is no single parameter set that works across cases and that binarisation and border sampling materially change results [4]. Mitigate by deriving thresholds from per-page statistics rather than hard-coding them, and by exposing the intermediate gap distributions so a bad page can be diagnosed rather than guessed at.

UVDoc was trained on Doc3D and the UVDoc dataset, so it may not generalise to surfaces unlike those, and it can distort a page that was already flat. Gate it: only apply unwarping when the page-corner quad or a flatness measure says the surface is deformed.

OCR-first inverts an assumption elsewhere in the pipeline. Today, a region that the detector misses is invisible to OCR. After the change, OCR sees the whole page, so text that previously fell outside a box now appears — including text you may not want, such as a stamp impression or a background logo. Expect to add filtering rather than to remove it.

Finally, CPU cost adds up when you stack unwarping, polygon detection and per-line rectification. Measure the end-to-end page time at each stage rather than at the end, so you know which stage to optimise.

## References

[1]: https://arxiv.org/html/1807.01544v2 "TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes"

[2]: https://arxiv.org/abs/2104.10442 "Fourier Contour Embedding for Arbitrary-Shaped Text Detection"

[3]: https://mmocr.readthedocs.io/en/v0.5.0/textdet_models.html "MMOCR Text Detection Models"

[4]: http://www.ipol.im/pub/art/2026/591/article.pdf "Voronoi Diagrams for Page Segmentation"

[5]: https://kraken.re/main/introduction_to_atr.html "Introduction to Automatic Text Recognition — Kraken"

[6]: https://github.com/tanguymagne/UVDoc "UVDoc: Neural Grid-based Document Unwarping"

[7]: https://developers.google.com/resources/api-libraries/documentation/vision/v1/java/latest/com/google/api/services/vision/v1/model/BoundingPoly.html "Cloud Vision API BoundingPoly"

[8]: https://www3.cs.stonybrook.edu/~cvl/docunet.html "DocUNet: Document Image Unwarping via a Stacked U-Net"

[9]: https://ieeexplore.ieee.org/document/9525302/ "ABCNet v2: Adaptive Bezier-Curve Network for Real-Time End-to-End Text Spotting"

[10]: https://arxiv.org/html/2302.02887v2 "UVDoc: Neural Grid-based Document Unwarping"

[11]: https://users.iit.demokritos.gr/~bgat/RLSA_values.pdf "Determination of run-length smoothing values for document segmentation"

[12]: https://arxiv.org/html/2504.10258v3 "XY-Cut++: Advanced Layout Ordering via Hierarchical Masked Attention"

[13]: https://link.springer.com/article/10.1007/s42979-023-01693-5 "Computer Vision Based Automatic Margin Computation Model for Digital Document Images"

[14]: https://github.com/bytedance/SPTSv2 "SPTS v2: Single-Point Scene Text Spotting"

[15]: https://github.com/fh2019ustc/DocGeoNet "DocGeoNet: Geometric Representation Learning for Document Image Rectification"