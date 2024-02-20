import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyCode;
import javafx.scene.text.Text;
import javafx.scene.text.TextAlignment;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.Group;
import javafx.animation.*;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.util.Duration;
import javafx.scene.layout.BorderPane;
import javafx.geometry.Pos;
//This was Pane at first but had to change it to Group
public class TitleGroup extends Group {
	private ImageView imageView = new ImageView(new Image("assets/favicon/1.png"));
	private Text text = new Text(40, 40, "PRESS ENTER TO START\nPRESS ESC TO EXIT");
	TitleGroup(){
		// text that will go thorugh phase tranformation
        text.setFont(Font.font(14*DuckHunt.SCALE));
        text.setFill(Color.ORANGE);
        text.setTextAlignment(TextAlignment.CENTER);
        BorderPane border = new BorderPane();
        //using border to make sure it is always in middle and does not change with SCALE
        border.setBottom(text);
        BorderPane.setAlignment(text, Pos.TOP_CENTER);
        border.setPrefSize(DuckHunt.SCALE*256, 200*DuckHunt.SCALE);
        
		imageView.setFitWidth(DuckHunt.SCALE*256);
		imageView.setFitHeight(DuckHunt.SCALE*240);
	    getChildren().addAll(imageView,border);
	    // Create a handler for changing text
	    EventHandler<ActionEvent> eventHandler = e -> {
	    	if (text.getText().length() != 0) {
	    			text.setText("");
	    	}else {
	    			text.setText("PRESS ENTER TO START\nPRESS ESC TO EXIT");
	    		}
	    	};
	    //Phase animation
	    Timeline animation = new Timeline(
	    		new KeyFrame(Duration.millis(300),eventHandler));
	    animation.setCycleCount(Timeline.INDEFINITE);
	    animation.play();
	    //Ability to go backward
	    this.setOnKeyPressed(e ->{

	    	if (e.getCode() == KeyCode.ENTER) {
	    		DuckHunt.getGameStateManager()
	    		.setCurrentState(GameStateManager.GameState.BACKGROUND);

	    	}
	    });
	    // bind the image view size to the pane size
	    this.requestFocus();
	}

}
