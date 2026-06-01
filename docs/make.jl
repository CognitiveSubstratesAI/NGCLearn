using Documenter
using NGCLearn

DocMeta.setdocmeta!(NGCLearn, :DocTestSetup, :(using NGCLearn); recursive=true)

makedocs(;
    modules=[NGCLearn],
    authors="CognitiveSubstrates AI",
    repo=Remotes.GitHub("CognitiveSubstratesAI", "NGCLearn"),
    sitename="NGCLearn.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cognitivesubstratesai.github.io/NGCLearn/stable/",
        edit_link="main",
        assets=String[]
    ),
    pages=[
        "Home" => "index.md",
        "Overview" => "overview.md",
        "Getting Started" => [
            "Installation" => "getting_started/installation.md",
            "Quickstart" => "getting_started/quickstart.md"
        ],
        "Components" => "components.md",
        "Modeling" => [
            "Neuronal Cells" => "modeling/neurons.md",
            "Synapses" => "modeling/synapses.md",
            "Input Encoders & Traces" => "modeling/input_encoders.md"
        ],
        "Models" => "models.md",
        "Model Museum" => [
            "Discriminative Predictive Coding" => "museum/pcn_discrim.md",
            "Diehl & Cook Spiking Network" => "museum/snn_dc.md"
        ],
        "Architecture & Design" => "architecture.md",
        "API Reference" => "api/index.md"
    ],
    warnonly=[:missing_docs, :cross_references]
)

deploydocs(;
    repo="github.com/CognitiveSubstratesAI/NGCLearn",
    devbranch="main"
)
