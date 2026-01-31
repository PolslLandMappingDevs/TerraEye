**TerraEye – flexible land mapping**

**Link to current demo:**
https://huggingface.co/spaces/PolslLandMappingDevs/TerraEye

TerraEye is an open-source project. It provides tools for mapping, spatial analysis,  
and visualization of geospatial and satellite data. The system combines a classical  
web application architecture with the TerraMind foundation model as an analytical layer,  
enabling satellite image processing, thematic map generation, and spatial analysis  
for environmental, planning, and research applications.



![Menu and an interactive globe view](https://cdn-uploads.huggingface.co/production/uploads/697a0786a0ea5d8c16529480/2bsVQ5HhknyRUDUBb2LJf.png) *Figure 1: Menu and an interactive globe view used for global navigation and selection of the area of interest.*




![Zoomed menu and an interactive globe view ](https://cdn-uploads.huggingface.co/production/uploads/697a0786a0ea5d8c16529480/6WUa6F8X-AL9C3QIjdEOV.png) *Figure 2: Zoomed menu and an interactive globe view used for global navigation and selection of the area of interest.*

---

**Why it’s interesting?**

Geospatial mapping and analysis are essential for modern information systems,  
including spatial planning, environmental monitoring, infrastructure change detection,  
and land-use assessment. With increasing availability of satellite data, there is growing  
demand for tools that not only visualize geospatial information but also automatically  
interpret it. TerraEye offers a flexible, modular, open-source platform that leverages  
foundation models to support advanced geospatial analysis, suitable for education,  
research, and potential industrial use.

---

**How It Works**

The system is modular: the backend handles data processing and TerraMind integration,  
while the frontend provides an interactive interface for maps and analysis results.  
TerraMind performs multimodal satellite image analysis, generates land-use,  
land-cover maps and supports spatial metric calculations.

---

**Advanced use of TerraMind**

TerraMind is the core analytical engine of the project. The model is trained for Earth  
observation tasks, allowing it to process satellite data alongside traditional spectral  
indices.

**Core TerraMind functions**

1. **Multimodal satellite analysis**  
The model processes satellite images and classifies pixels according to land  
cover and land use, such as forests, water, urban areas, and croplands.

2. **Semantic map generation**  
Model outputs are used to create LULC maps that support both visual  
interpretation and further spatial calculations.

3. **Context-aware analysis**  
By providing semantic labels for pixels, TerraMind enhances spectral analyses  
e.g., NDVI (vegetation), NDWI (water), NDBI (built-up areas) enabling  
interpretation of indices in the context of land type.

---

**Spectral indices**

Spectral indices are mathematical combinations of satellite spectral bands designed  
to highlight specific landscape properties, such as vegetation health, soil brightness,  
or water presence. In TerraEye, model outputs can be combined with spectral index  
calculations to produce richer analytical layers.


![Comparison of two TerraMind variants](https://cdn-uploads.huggingface.co/production/uploads/697a0786a0ea5d8c16529480/sWFYELUGHaizzIb72UFZK.png) *Figure 3: Comparison of two TerraMind variants (e.g., Large and Tiny), illustrating the progression from RGB satellite  imagery to TerraMind outputs enhanced with spectral indices.*

---

**Evaluation and Model Comparison**

To assess different TerraMind variants and spectral calculations, the system uses  
standard pixel-based metrics:

- **Pixel Accuracy** – proportion of „correctly” classified pixels  
- **Precision / Recall** – quality of predictions for individual classes  
- **Dice Score** – balanced measure for small or sparse classes  
- **Intersection over Union (IoU, mIoU, fwIoU)** – overlap quality, including class  
  weighting  


![Metrics calculated between two sets](https://cdn-uploads.huggingface.co/production/uploads/697a0786a0ea5d8c16529480/W9om_5ZW2Aauw4T4kuKsK.png) *Figure 4: evaluation metrics calculated between two sets of segmentation maps produced by different TerraMind  variants (e.g., Large and Tiny), where one variant serves as the reference (“ground truth”) delineation.*

These metrics allow comparison of model variants in terms of accuracy  
and computational cost, critical for real-world scenarios and potential on-board  
satellite processing.

---

**Demo Features**

The current demo offers:

- Interactive visualization of geospatial data  
- Presentation of TerraMind-generated analysis results  
- Basic spatial metric calculations  
- Real-time communication between frontend and backend  

---

**Lessons Learned**

The project provided practical experience in integrating AI models with GIS systems,  
handling satellite data, and evaluating model outputs. Despite no prior experience in AI  
or satellite analysis, the team quickly adapted, highlighting the value of modular design,  
practical problem-solving, and interdisciplinary teamwork.

---

**Team**

▪ Authors: Jakub Bodzioch, Łukasz Wolf, Kajetan Hołdan, Jan Piechota,  
Michał Figołuszka, Karol Bieżuński  

▪ Affiliation: Silesian University of Technology, Poland 

▪ Supervisor: Jakub Nalepa, PhD, DSc  

▪ Contact: jb305900@student.polsl.pl
