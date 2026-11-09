// Campos que precisam ser enviados como NÚMERO INTEIRO
// (o formulário entrega tudo como texto; a API espera números)
const CAMPOS_INTEIROS = ["idade", "pressao_repouso", "colesterol",
                         "glicemia_jejum_alta", "freq_cardiaca_max"];

// Campo que precisa ser enviado como número DECIMAL
const CAMPOS_DECIMAIS = ["depressao_st"];

// Pega o formulário e a área de resultado pelo id
const formulario = document.getElementById("formulario");
const resultado = document.getElementById("resultado");


// Mostra uma mensagem simples (carregando ou erro) dentro de um cartão
function mostrarMensagem(texto, classe) {

    // innerHTML troca todo o conteúdo da área de resultado pelo HTML abaixo
    resultado.innerHTML = '<div class="cartao ' + classe + '">' + texto + '</div>';
}


// Monta o cartão de resultado a partir da resposta da API
function mostrarResultado(dados) {

    // Classe CSS conforme o risco: "alto" (vermelho) ou "baixo" (verde)
    const classe = dados.risco === "ALTO" ? "alto" : "baixo";

    // Monta o HTML do cartão; os valores vêm da resposta da API
    resultado.innerHTML =
        '<div class="cartao ' + classe + '">' +

            // Título com o risco
            '<div class="risco-titulo">Risco ' + dados.risco + '</div>' +

            // Probabilidade em destaque
            '<div class="probabilidade">' + dados.probabilidade_doenca + '%</div>' +
            '<div class="legenda">probabilidade estimada de doença cardíaca</div>' +

            // Barra: o preenchimento tem a largura da probabilidade;
            // a marca vertical mostra onde fica o limiar de 30%
            '<div class="barra">' +
                '<div class="barra-preenchida" style="width: ' + dados.probabilidade_doenca + '%"></div>' +
                '<div class="marca-limiar" style="left: ' + dados.limiar_usado + '%"></div>' +
            '</div>' +
            '<div class="legenda">a linha preta marca o limiar de ' + dados.limiar_usado + '%</div>' +

            // Frase de interpretação e aviso
            '<p>' + dados.interpretacao + '</p>' +
            '<p class="aviso">' + dados.aviso + '</p>' +

        '</div>';
}


// "Escuta" o envio do formulário (clique em "Calcular risco")
formulario.addEventListener("submit", async function (evento) {

    // Impede o comportamento padrão (recarregar a página)
    evento.preventDefault();

    // Lê todos os campos do formulário
    const dadosFormulario = new FormData(formulario);

    // Objeto que será enviado para a API
    const paciente = {};

    // Percorre cada campo: nome (ex: "idade") e valor digitado (texto)
    for (const [campo, valor] of dadosFormulario.entries()) {

        if (CAMPOS_INTEIROS.includes(campo)) {
            // Converte texto em número inteiro (ex: "54" -> 54)
            paciente[campo] = parseInt(valor, 10);
        } else if (CAMPOS_DECIMAIS.includes(campo)) {
            // Converte texto em número decimal (ex: "1.5" -> 1.5)
            paciente[campo] = parseFloat(valor);
        } else {
            // Campos de texto seguem como estão (ex: "M", "ASY")
            paciente[campo] = valor;
        }
    }

    // Mostra uma mensagem enquanto espera a resposta da API
    mostrarMensagem("Calculando...", "carregando");

    try {
        // Envia os dados para a rota /prever da própria API (POST com JSON)
        const resposta = await fetch("/prever", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(paciente)
        });

        // Converte a resposta da API (JSON) em objeto JavaScript
        const dados = await resposta.json();

        // resposta.ok é falso quando a API recusa os dados (ex: erro 422)
        if (!resposta.ok) {
            mostrarMensagem("Dados inválidos: confira os campos e tente novamente.", "erro");
            return;
        }

        // Monta o cartão colorido com o resultado
        mostrarResultado(dados);

        // Rola a tela até o resultado (útil no celular, onde o formulário é longo)
        resultado.scrollIntoView({ behavior: "smooth" });

    } catch (erro) {
        // Cai aqui se a API estiver desligada ou sem conexão
        mostrarMensagem("Não foi possível falar com a API. Ela está ligada?", "erro");
    }
});
