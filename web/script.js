// Campos que precisam ser enviados como NÚMERO INTEIRO
// (o formulário entrega tudo como texto; a API espera números).
// A pressão não está aqui: ela tem tratamento próprio (função converterPressao)
const CAMPOS_INTEIROS = ["idade", "colesterol",
                         "glicemia_jejum_alta", "freq_cardiaca_max"];

// Campo que precisa ser enviado como número DECIMAL
const CAMPOS_DECIMAIS = ["depressao_st"];

// Pega o formulário e a área de resultado pelo id
const formulario = document.getElementById("formulario");
const resultado = document.getElementById("resultado");

// Pega o campo de pressão (validado à parte, por aceitar o formato "12x8")
const campoPressao = document.getElementById("pressao_repouso");


// Converte a pressão no padrão brasileiro para mm Hg, como a API espera.
// Aceita "12x8", "12/8", "12 x 8", "13,5x9", "120x80" ou só "12".
// Devolve o número em mm Hg (ex: 120) ou null se o texto não for válido.
function converterPressao(texto) {

    // Procura o PRIMEIRO número do texto (a pressão máxima), com ou sem decimal.
    // ^\s*      -> início do texto, ignorando espaços
    // \d{1,3}   -> de 1 a 3 dígitos (ex: 12 ou 120)
    // ([.,]\d)? -> decimal opcional com ponto ou vírgula (ex: 13,5)
    const achado = texto.match(/^\s*(\d{1,3}([.,]\d)?)/);

    // Nenhum número no começo do texto: formato inválido
    if (achado === null) {
        return null;
    }

    // Troca vírgula por ponto ("13,5" -> "13.5") e converte em número
    let maxima = parseFloat(achado[1].replace(",", "."));

    // Abaixo de 30, está no formato curto brasileiro (12 = 120 mm Hg): multiplica por 10
    if (maxima < 30) {
        maxima = maxima * 10;
    }

    // Arredonda para inteiro (a API espera número inteiro: 13,5 -> 135)
    maxima = Math.round(maxima);

    // Fora dos limites aceitos pela API (50 a 250 mm Hg): inválido
    if (maxima < 50 || maxima > 250) {
        return null;
    }

    // Valor pronto para enviar
    return maxima;
}


// Sempre que a pessoa digitar na pressão, limpa o aviso de erro anterior
campoPressao.addEventListener("input", function () {
    // setCustomValidity("") = campo volta a ser considerado válido
    campoPressao.setCustomValidity("");
});


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

        if (campo === "pressao_repouso") {
            // Pressão: converte "12x8" em 120 (mm Hg)
            paciente[campo] = converterPressao(valor);
        } else if (CAMPOS_INTEIROS.includes(campo)) {
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

    // Pressão em formato que não deu para entender: avisa no próprio campo e não envia
    if (paciente.pressao_repouso === null) {
        // Mensagem que o navegador mostra no balão de erro do campo
        campoPressao.setCustomValidity("Digite a pressão no formato 12x8 (ou 120x80).");
        // Mostra o balão e coloca o cursor no campo
        campoPressao.reportValidity();
        // Para aqui: nada é enviado para a API
        return;
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

        // 429 = o Nginx barrou por excesso de pedidos seguidos (limite de requisições).
        // Verificado ANTES de ler o JSON, porque nesse caso o Nginx responde em HTML, não em JSON
        if (resposta.status === 429) {
            mostrarMensagem("Muitas tentativas seguidas. Aguarde alguns segundos e tente de novo.", "erro");
            return;
        }

        // resposta.ok é falso quando a API recusa os dados (ex: erro 422)
        if (!resposta.ok) {
            mostrarMensagem("Dados inválidos: confira os campos e tente novamente.", "erro");
            return;
        }

        // Converte a resposta da API (JSON) em objeto JavaScript
        const dados = await resposta.json();

        // Monta o cartão colorido com o resultado
        mostrarResultado(dados);

        // Rola a tela até o resultado (útil no celular, onde o formulário é longo)
        resultado.scrollIntoView({ behavior: "smooth" });

    } catch (erro) {
        // Cai aqui se a API estiver desligada ou sem conexão
        mostrarMensagem("Não foi possível falar com a API. Ela está ligada?", "erro");
    }
});
