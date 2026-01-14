# System Design - Group Submission

This folder contains group system design diagrams following UML standards.

## Requirements

### File Format
- **Format:** draw.io XML (`.drawio`) and exported PDF (.pdf)
- **Naming:** `YYYY-FamilyName1-FamilyName2-FamilyName3.drawio` (alphabetical order, ASCII only)
- **Standard:** UML (Unified Modeling Language)
- **License:** Include license in diagram metadata or as text element (CC-BY-4.0 recommended)
- **Length:** Maximum 1 A4 page equivalent when exported

---

## Content Requirements

Your system design diagram must illustrate your AI agent architecture using UML notation. Include:

| Element | Description |
|---------|-------------|
| **Components** | Major system components (agents, tools, data stores) |
| **Interfaces** | How components communicate |
| **Data Flow** | Direction of data/control flow |
| **External Systems** | APIs, databases, or services your system uses |

### Recommended UML Diagram Types

Choose the most appropriate for your system:

1. **Component Diagram** - Shows system components and their relationships (most common)
2. **Sequence Diagram** - Shows interaction flow over time
3. **Activity Diagram** - Shows workflow/process steps
4. **Class Diagram** - Shows classes and their relationships (if OOP-based)

---

## Grading Criteria

Deliverable scored as passed (1) if handed in with acceptable quality before the end of the course.

---

## UML Notation

Add link to latest reference document.

### Component Diagram Elements

```
┌─────────────────┐
│  <<component>>  │     Component (with stereotype)
│   ComponentName │
└─────────────────┘

┌─────────────────┐
│ ComponentName   │     Component (simple)
└─────────────────┘

    ○──────────────     Provided interface (lollipop)

    ○──┐
       │                Required interface (socket)
    ───┘

─────────────────►      Dependency arrow

════════════════►      Data flow (thick arrow)
```

### Common Stereotypes for AI Agents

| Stereotype | Use for |
|------------|---------|
| `<<agent>>` | AI agent components |
| `<<tool>>` | Agent tools/capabilities |
| `<<model>>` | ML models |
| `<<database>>` | Data storage |
| `<<api>>` | External API interfaces |
| `<<service>>` | External services |

---

## Example System Design

### Component Diagram for HRV Analysis Agent

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         HRV Analysis Agent System                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐            │
│  │ <<agent>>    │      │ <<tool>>     │      │ <<tool>>     │            │
│  │              │─────▶│              │─────▶│              │            │
│  │ Orchestrator │      │ ECG Loader   │      │ Signal       │            │
│  │              │      │              │      │ Processor    │            │
│  └──────┬───────┘      └──────────────┘      └──────┬───────┘            │
│         │                                           │                     │
│         │                                           ▼                     │
│         │              ┌──────────────┐      ┌──────────────┐            │
│         │              │ <<model>>    │      │ <<tool>>     │            │
│         │              │              │◀─────│              │            │
│         │              │ Classifier   │      │ Feature      │            │
│         │              │              │      │ Extractor    │            │
│         │              └──────┬───────┘      └──────┬───────┘            │
│         │                     │                     │                     │
│         │                     │    ┌────────────────┘                     │
│         │                     │    │  (features + prediction)             │
│         │                     ▼    ▼                                      │
│         │              ┌──────────────┐      ┌──────────────┐            │
│         │              │ <<tool>>     │      │ <<artifact>> │            │
│         └─────────────▶│              │─────▶│              │            │
│                        │ Report       │      │ PDF Report   │            │
│                        │ Generator    │      │              │            │
│                        └──────────────┘      └──────────────┘            │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘

External:
┌──────────────┐
│ <<database>> │
│ File System  │  ECG Data Files (.txt)
└──────────────┘
```

---

## How to Use draw.io

### Option 1: Online (Recommended)

1. Go to https://app.diagrams.net (draw.io)
2. Choose "Create New Diagram"
3. Select storage location (Device for local file)
4. Choose "UML" from template categories or start blank
5. Use the UML shape library from the left panel
6. Save as `.drawio` file

### Option 2: Desktop Application

Download from: https://www.drawio.com/

### Creating Your Diagram

1. **Open draw.io**
2. **Enable UML shapes:**
   - Click "More Shapes" at bottom of left panel
   - Check "UML" category
   - Click "Apply"
3. **Add components:**
   - Drag shapes from left panel
   - Double-click to edit labels
4. **Add connections:**
   - Hover over shape edge until blue arrows appear
   - Drag to target shape
5. **Add stereotypes:**
   - Add text `<<stereotype>>` above component name
6. **Save:**
   - File → Save As
   - Name: `YYYY-FamilyName1-FamilyName2-FamilyName3.drawio`

---

## Best Practices

### Layout
- Arrange components logically (input → processing → output)
- Use consistent spacing
- Align components in a grid
- Keep the diagram on a single page

### Labels
- Use clear, descriptive names
- Include stereotypes for clarity
- Label all connections/interfaces

### Style
- Use consistent colors (or keep default)
- Make text large enough to read
- Use standard UML notation

---

## Including License in draw.io

Add a text element to your diagram:

```
License: CC-BY-4.0
Authors: Chen Wei, Lin MeiLing, Wang XiaoMing
```

Or set in File → Properties → Description.

---

## Exporting for Review

To share a visual preview:

1. File → Export as → PNG or PDF
2. Name: `YYYY-FamilyName1-FamilyName2-FamilyName3.pdf`
3. Include both `.drawio` (source) and `.pdf` (preview) in submission

---

## Submission Checklist

Before submitting, verify:

- [ ] File named correctly: `YYYY-Name1-Name2-Name3.drawio` (alphabetical)
- [ ] Only ASCII characters in filename
- [ ] License declaration included
- [ ] Uses standard UML notation
- [ ] All major components represented
- [ ] Connections and data flow clearly shown
- [ ] Labels are readable
- [ ] Fits on approximately 1 A4 page
- [ ] No personally identifiable information (PII)
- [ ] All group members listed
