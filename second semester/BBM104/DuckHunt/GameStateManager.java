import javafx.beans.property.ReadOnlyObjectWrapper;

public class GameStateManager {
    public enum GameState {
        TITLE,
        BACKGROUND,
        LEVEL1,
        LEVEL2,
        LEVEL3,
        LEVEL4,
        LEVEL5,
        LEVEL6
    }

    private ReadOnlyObjectWrapper<GameState> currentState;

    public GameStateManager() {
        currentState = new ReadOnlyObjectWrapper<>(GameState.TITLE);
    }

    public GameState getCurrentState() {
        return currentState.get();
    }

    public ReadOnlyObjectWrapper<GameState> currentStateProperty() {
        return currentState;
    }

    public void setCurrentState(GameState state) {
        currentState.set(state);
    }
}