# LoRA example gallery

These 30 images are curated previews of historical 4K WaveAlign generations with FLUX.1-dev and a LoRA adapter. Images 01–20 were selected from a previously reviewed set of 50; images 21–30 add library, architecture, and aerial subjects from a separate LoRA run. Every selected source has a completed run record and an image hash matching its original PNG.

The historical runs used the **old coupled-noise initialization and linear guidance release**. The public CLI and demo retain the old noise as their default but now default to **cosine** release. The pictures below therefore document the historical linear setting; they are not examples of the cosine default, and they are not a benchmark comparison. Reproducing an exact original also requires the same model revision, adapter weights and conversion, and software environment.

The gallery contains only JPEG previews, scaled to fit within 1600 × 1600 pixels without cropping. The original 4K PNGs and third-party model and LoRA weights are not included. [manifest.json](manifest.json) records each prompt, seed, LoRA identifier, scale and SHA-256, original dimensions and SHA-256, source job/key, and preview dimensions and SHA-256. Adapter identifiers are provenance labels, not download instructions.

## Selected 20 from the original 50

| | | |
|:---:|:---:|:---:|
| [![Mountain traveler](images/01-mountain-traveler.jpg)](images/01-mountain-traveler.jpg)<br>01 · Mountain traveler | [![Wildflower portrait](images/02-wildflower-portrait.jpg)](images/02-wildflower-portrait.jpg)<br>02 · Wildflower portrait | [![Mountain guide](images/03-mountain-guide.jpg)](images/03-mountain-guide.jpg)<br>03 · Mountain guide |
| [![Red fox](images/04-red-fox.jpg)](images/04-red-fox.jpg)<br>04 · Red fox | [![River bear](images/05-river-bear.jpg)](images/05-river-bear.jpg)<br>05 · River bear | [![Kingfisher](images/06-kingfisher.jpg)](images/06-kingfisher.jpg)<br>06 · Kingfisher |
| [![Forest waterfall](images/07-forest-waterfall.jpg)](images/07-forest-waterfall.jpg)<br>07 · Forest waterfall | [![Coastal cliffs](images/08-coastal-cliffs.jpg)](images/08-coastal-cliffs.jpg)<br>08 · Coastal cliffs | [![Greenhouse botanist](images/09-greenhouse-botanist.jpg)](images/09-greenhouse-botanist.jpg)<br>09 · Greenhouse botanist |
| [![Violin maker](images/10-violin-maker.jpg)](images/10-violin-maker.jpg)<br>10 · Violin maker | [![Orca](images/11-orca.jpg)](images/11-orca.jpg)<br>11 · Orca | [![Hummingbird](images/12-hummingbird.jpg)](images/12-hummingbird.jpg)<br>12 · Hummingbird |
| [![Puffin](images/13-puffin.jpg)](images/13-puffin.jpg)<br>13 · Puffin | [![Sahara caravan](images/14-sahara-caravan.jpg)](images/14-sahara-caravan.jpg)<br>14 · Sahara caravan | [![Redwood creek](images/15-redwood-creek.jpg)](images/15-redwood-creek.jpg)<br>15 · Redwood creek |
| [![Rice terraces](images/16-rice-terraces.jpg)](images/16-rice-terraces.jpg)<br>16 · Rice terraces | [![Aurora cabin](images/17-aurora-cabin.jpg)](images/17-aurora-cabin.jpg)<br>17 · Aurora cabin | [![Atlantic coast](images/18-atlantic-coast.jpg)](images/18-atlantic-coast.jpg)<br>18 · Atlantic coast |
| [![Teacup village](images/19-teacup-village.jpg)](images/19-teacup-village.jpg)<br>19 · Teacup village | [![Miniature coastal village](images/20-miniature-coastal-village.jpg)](images/20-miniature-coastal-village.jpg)<br>20 · Miniature coastal village | |

## Ten additional subjects from another LoRA run

| | | |
|:---:|:---:|:---:|
| [![Grand library](images/21-grand-library.jpg)](images/21-grand-library.jpg)<br>21 · Grand library | [![Waterfront opera house](images/22-waterfront-opera.jpg)](images/22-waterfront-opera.jpg)<br>22 · Waterfront opera house | [![Reading room](images/23-reading-room.jpg)](images/23-reading-room.jpg)<br>23 · Reading room |
| [![Coastal town aerial](images/24-coastal-town-aerial.jpg)](images/24-coastal-town-aerial.jpg)<br>24 · Coastal town aerial | [![Railway station](images/25-railway-station.jpg)](images/25-railway-station.jpg)<br>25 · Railway station | [![Agricultural aerial](images/26-agricultural-aerial.jpg)](images/26-agricultural-aerial.jpg)<br>26 · Agricultural aerial |
| [![Art museum](images/27-art-museum.jpg)](images/27-art-museum.jpg)<br>27 · Art museum | [![Mountain reservoir aerial](images/28-mountain-reservoir-aerial.jpg)](images/28-mountain-reservoir-aerial.jpg)<br>28 · Mountain reservoir aerial | [![City square aerial](images/29-city-square-aerial.jpg)](images/29-city-square-aerial.jpg)<br>29 · City square aerial |
| [![Volcanic caldera aerial](images/30-volcanic-caldera-aerial.jpg)](images/30-volcanic-caldera-aerial.jpg)<br>30 · Volcanic caldera aerial | | |
